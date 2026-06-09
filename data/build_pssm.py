"""build_pssm.py — 자연 항체 CDR 위치별 분포(q_target) 구축. (abnum env 전용: anarci 필요)

설계 CDR 위치(config 정의) → ANARCI IMGT 넘버링으로 IMGT 키 매핑 → 레퍼런스 항체(TheraSAbDab 등)
들에서 *같은 IMGT 위치*의 AA를 모아 위치별 빈도(q_pos) + 전역(q_global) 구축 → npz 저장.
→ 런타임(esmfold2)은 anarci 없이 npz만 로드 (composition.py).

설계 CDR 인덱스 순서는 run.py 와 동일 규칙(heavy CDR ranges → light, 각 range(s,e), global 오름차순).

실행 (abnum env):
  PATH=/home/kyeongtak/.conda/envs/abnum/bin:$PATH \
  /home/kyeongtak/.conda/envs/abnum/bin/python build_pssm.py \
      --config configs/trastuzumab_her2.yaml \
      --refs /home/aidx/DB/AGAB_MSADB/TheraSAbDab_SeqStruc_OnlineDownload.csv \
      --out data/trastuzumab_qtarget.npz
"""
from __future__ import annotations

import argparse
import csv
import os
import warnings

import numpy as np
import yaml

warnings.filterwarnings("ignore")
from anarci import anarci  # noqa: E402  (abnum env)

AA = "ARNDCQEGHILKMFPSTWYV"          # 20-AA 표준 순서 (composition.py 가 j-순서로 재정렬)
AA_IDX = {a: i for i, a in enumerate(AA)}
# IMGT CDR 정의 (IMGT 넘버링): CDR1 27-38, CDR2 56-65, CDR3 105-117
IMGT_CDR = {1: (27, 38), 2: (56, 65), 3: (105, 117)}
LINKER = "GGGGSGGGGSGGGGS"


def _clean(s):
    return s.replace(" ", "").replace("\n", "").replace("\r", "").upper()


def number_seq(seq):
    """seq → {(imgt_num, inscode): AA}, chain_type ('H'/'K'/'L').  실패 시 (None,None)."""
    try:
        numbered, details, _ = anarci([("q", seq)], scheme="imgt", output=False)
    except Exception:
        return None, None
    if not numbered or numbered[0] is None:
        return None, None
    dom = numbered[0][0][0]
    ct = details[0][0]["chain_type"]
    m = {(num, ins.strip()): aa for (num, ins), aa in dom if aa != "-"}
    return m, ct


def design_imgt_keys(cfg):
    """config 의 설계 CDR 위치(run.py 와 동일 순서)를 IMGT 키로 매핑.
    반환: keys = [(chain_type, (imgt_num, ins), cdr_name), ...]  (cdr_idx 순서와 1:1)."""
    vh = _clean(cfg["antibody"]["heavy"]["vh_sequence"])
    vl = _clean(cfg["antibody"]["light"]["vl_sequence"])

    def vdom_keys(seq, cdr_ranges, want_ct):
        m_num, ct = number_seq(seq)
        if m_num is None:
            raise RuntimeError("설계 항체 넘버링 실패")
        # seq 인덱스 → IMGT 키 (numbered 순서가 seq 순서)
        numbered, _, _ = anarci([("q", seq)], scheme="imgt", output=False)
        dom = numbered[0][0][0]
        idx2key, si = {}, 0
        for (num, ins), aa in dom:
            if aa == "-":
                continue
            idx2key[si] = (num, ins.strip())
            si += 1
        out = {}  # cdr_name -> list of (seq_idx, imgt_key)
        for name, (s, e) in cdr_ranges.items():
            out[name] = [(p, idx2key[p]) for p in range(s, e) if p in idx2key]
        return out, ct

    h_map, h_ct = vdom_keys(vh, cfg["antibody"]["heavy"]["cdr_ranges"], "H")
    l_map, l_ct = vdom_keys(vl, cfg["antibody"]["light"]["cdr_ranges"], "L")

    # run.py 순서: heavy 전체(global 오름차순) → light(global 오름차순). scFv 내 global = idx (heavy), voff+idx (light)
    voff = len(vh) + len(LINKER)
    rows = []  # (global_pos, chain_type, imgt_key, cdr_name)
    for name, lst in h_map.items():
        for p, key in lst:
            rows.append((p, h_ct, key, f"H{name[-1]}"))
    for name, lst in l_map.items():
        for p, key in lst:
            rows.append((voff + p, l_ct, key, f"L{name[-1]}"))
    rows.sort(key=lambda r: r[0])      # global 오름차순 = run.py cdr_idx 순서
    return [(ct, key, name) for _, ct, key, name in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/trastuzumab_her2.yaml")
    ap.add_argument("--refs", nargs="+",
                    default=["/home/aidx/DB/AGAB_MSADB/TheraSAbDab_SeqStruc_OnlineDownload.csv"])
    ap.add_argument("--out", default="data/trastuzumab_qtarget.npz")
    ap.add_argument("--pseudo", type=float, default=2.0, help="Dirichlet pseudocount")
    ap.add_argument("--max-refs", type=int, default=80000,
                    help="넘버링할 레퍼런스 서열 상한 (subsample; 0=전체)")
    ap.add_argument("--ncpu", type=int, default=8, help="ANARCI/hmmscan 병렬 코어 수")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    keys = design_imgt_keys(cfg)       # cdr_idx 순서의 (chain_type, imgt_key, cdr_name)
    n = len(keys)
    print(f"[pssm] 설계 CDR 위치 {n}개 (IMGT 매핑 완료)")
    by_chain = {}                      # chain_type -> {imgt_key: 위치 인덱스 목록}
    for i, (ct, key, _name) in enumerate(keys):
        by_chain.setdefault(ct, {}).setdefault(key, []).append(i)

    counts = np.zeros((n, 20), dtype=np.float64)
    gcounts = np.zeros(20, dtype=np.float64)   # 전역(모든 CDR 잔기)
    nseq = 0

    # 레퍼런스 항체 VH/VL 수집
    def iter_ref_seqs(path):
        if path.endswith(".csv"):
            with open(path, encoding="utf-8-sig") as f:
                rd = csv.DictReader(f)
                cols = rd.fieldnames or []
                # OAS compact(vdomain_aa) vs TheraSAbDab(HeavySequence/LightSequence) 자동 감지
                seq_cols = (["vdomain_aa"] if "vdomain_aa" in cols
                            else ["HeavySequence", "LightSequence"])
                for row in rd:
                    for col in seq_cols:
                        s = _clean(row.get(col, "") or "")
                        if 70 <= len(s) <= 200 and set(s) <= set(AA):
                            yield s
        else:  # fasta
            seq = ""
            for line in open(path):
                if line.startswith(">"):
                    if seq:
                        yield _clean(seq)
                    seq = ""
                else:
                    seq += line.strip()
            if seq:
                yield _clean(seq)

    # 1) 레퍼런스 서열 수집 + subsample (numbering 이 느려 상한)
    import random
    random.seed(0)
    seqs = []
    for path in args.refs:
        if not os.path.exists(path):
            print(f"[pssm] WARN 없음: {path}"); continue
        s = list(iter_ref_seqs(path))
        print(f"[pssm] {path}: {len(s)} 서열")
        seqs += s
    if args.max_refs and len(seqs) > args.max_refs:
        seqs = random.sample(seqs, args.max_refs)
    print(f"[pssm] numbering 대상 {len(seqs)} 서열 (batch ANARCI)")

    def add_counts(m_num, ct):
        if ct not in by_chain:
            return False
        hit = False
        for key, idxs in by_chain[ct].items():
            aa = m_num.get(key)
            if aa and aa in AA_IDX:
                for i in idxs:
                    counts[i, AA_IDX[aa]] += 1
                hit = True
        for (num, ins), aa in m_num.items():
            if aa in AA_IDX and any(lo <= num <= hi for lo, hi in IMGT_CDR.values()):
                gcounts[AA_IDX[aa]] += 1
        return hit

    # 2) 배치 넘버링 (anarci 에 리스트 한꺼번에 → hmmscan 호출 1회/배치)
    B = 4000
    for i in range(0, len(seqs), B):
        batch = [(str(j), s) for j, s in enumerate(seqs[i:i + B])]
        try:
            numbered, details, _ = anarci(batch, scheme="imgt", output=False, ncpu=args.ncpu)
        except Exception as e:
            print(f"[pssm] batch {i} 실패: {e}"); continue
        for k in range(len(batch)):
            if numbered[k] is None:
                continue
            dom = numbered[k][0][0]
            ct = details[k][0]["chain_type"]
            m_num = {(num, ins.strip()): aa for (num, ins), aa in dom if aa != "-"}
            if add_counts(m_num, ct):
                nseq += 1
        print(f"[pssm] numbered {min(i + B, len(seqs))}/{len(seqs)} (기여 {nseq})")

    # 정규화 + pseudocount
    q_global = (gcounts + args.pseudo) / (gcounts.sum() + 20 * args.pseudo)
    q_pos = (counts + q_global * args.pseudo * 5) / \
            (counts.sum(1, keepdims=True) + args.pseudo * 5)   # 위치 데이터 빈약시 q_global 로 shrink
    support = counts.sum(1)             # 위치별 데이터 수

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, q_pos=q_pos.astype(np.float32), q_global=q_global.astype(np.float32),
             aa_order=AA, support=support.astype(np.int32),
             cdr_names=np.array([name for _, _, name in keys]))
    print(f"[pssm] 저장 → {args.out}  (위치 {n}, 기여서열 {nseq}, 평균 support {support.mean():.0f})")
    # 요약: 방향족 분율
    arom = [AA_IDX[a] for a in "WYF"]
    print(f"[pssm] q_global 방향족(W+Y+F) = {q_global[arom].sum():.3f}  "
          f"(Tyr={q_global[AA_IDX['Y']]:.3f} Trp={q_global[AA_IDX['W']]:.3f} Phe={q_global[AA_IDX['F']]:.3f})")


if __name__ == "__main__":
    main()
