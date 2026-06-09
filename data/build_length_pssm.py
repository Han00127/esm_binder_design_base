"""build_length_pssm.py — 길이층화 PSSM 사전구축. (abnum env: anarci)

자연 항체(OAS+TheraSAbDab) → ANARCI(IMGT) → 6개 CDR(H1..L3) 각각의
  (a) 길이 분포  length_hist[CDR][length]
  (b) 길이별 PSSM  pssm[CDR][length] = [length, 20]   (위치별 자연 AA 분포)
을 집계 → npz + json 저장. (trajectory 별 가변길이 설계에서 q 로 사용 예정.)

실행 (abnum env):
  PATH=/home/kyeongtak/.conda/envs/abnum/bin:$PATH \
  /home/kyeongtak/.conda/envs/abnum/bin/python build_length_pssm.py \
    --refs /home/aidx/DB/AGAB_MSADB/TheraSAbDab_SeqStruc_OnlineDownload.csv data/oas_paired_vdomains.csv \
    --out data/length_pssm --max-refs 120000 --ncpu 14
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import warnings
from collections import defaultdict

import numpy as np

warnings.filterwarnings("ignore")
from anarci import anarci  # noqa: E402

AA = "ARNDCQEGHILKMFPSTWYV"
AA_IDX = {a: i for i, a in enumerate(AA)}
IMGT_CDR = {1: (27, 38), 2: (56, 65), 3: (105, 117)}   # IMGT CDR1/2/3 번호범위
LINKER = "GGGGSGGGGSGGGGS"


def _clean(s):
    return s.replace(" ", "").replace("\n", "").replace("\r", "").upper()


def iter_ref_seqs(path):
    if path.endswith(".csv"):
        with open(path, encoding="utf-8-sig") as f:
            rd = csv.DictReader(f)
            cols = rd.fieldnames or []
            seq_cols = (["vdomain_aa"] if "vdomain_aa" in cols
                        else ["HeavySequence", "LightSequence"])
            for row in rd:
                for c in seq_cols:
                    s = _clean(row.get(c, "") or "")
                    if 70 <= len(s) <= 200 and set(s) <= set(AA):
                        yield s
    else:
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


def extract_cdrs(dom, ct):
    """numbered domain → {cdr_name: seq}. ct: 'H'|'K'|'L'."""
    pre = "H" if ct == "H" else "L"
    out = {}
    for c, (lo, hi) in IMGT_CDR.items():
        res = [aa for (num, ins), aa in dom if lo <= num <= hi and aa != "-"]
        if res:
            out[f"{pre}{c}"] = "".join(res)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", nargs="+", required=True)
    ap.add_argument("--out", default="data/length_pssm")
    ap.add_argument("--max-refs", type=int, default=120000)
    ap.add_argument("--ncpu", type=int, default=14)
    ap.add_argument("--pseudo", type=float, default=1.0)
    ap.add_argument("--min-support", type=int, default=30, help="(CDR,length) 최소 항체 수")
    args = ap.parse_args()

    # 수집 + subsample
    random.seed(0)
    seqs = []
    for p in args.refs:
        if not os.path.exists(p):
            print(f"WARN 없음: {p}"); continue
        s = list(iter_ref_seqs(p)); print(f"{p}: {len(s)} 서열"); seqs += s
    if args.max_refs and len(seqs) > args.max_refs:
        seqs = random.sample(seqs, args.max_refs)
    print(f"numbering 대상 {len(seqs)} 서열")

    length_hist = defaultdict(lambda: defaultdict(int))           # cdr -> length -> count
    counts = defaultdict(dict)                                    # cdr -> length -> [length,20]
    AROM = set("WYF")
    arom_hist = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # cdr->len->k(방향족수)->cnt
    wyf = defaultdict(lambda: defaultdict(lambda: np.zeros(3)))   # cdr->len->[W,Y,F] 누적
    nseq = 0
    B = 4000
    for i in range(0, len(seqs), B):
        batch = [(str(j), s) for j, s in enumerate(seqs[i:i + B])]
        try:
            numbered, details, _ = anarci(batch, scheme="imgt", output=False, ncpu=args.ncpu)
        except Exception as e:
            print(f"batch {i} 실패: {e}"); continue
        for k in range(len(batch)):
            if numbered[k] is None:
                continue
            dom = numbered[k][0][0]
            ct = details[k][0]["chain_type"]
            for name, seq in extract_cdrs(dom, ct).items():
                L = len(seq)
                length_hist[name][L] += 1
                if L not in counts[name]:
                    counts[name][L] = np.zeros((L, 20), dtype=np.float64)
                karom = 0
                for pos, aa in enumerate(seq):
                    if aa in AA_IDX:
                        counts[name][L][pos, AA_IDX[aa]] += 1
                    if aa in AROM:
                        karom += 1
                arom_hist[name][L][karom] += 1
                wyf[name][L] += [seq.count("W"), seq.count("Y"), seq.count("F")]
            nseq += 1
        print(f"numbered {min(i + B, len(seqs))}/{len(seqs)} (기여 {nseq})")

    # PSSM 정규화(pseudocount) + 저장 (min_support 이상만)
    arrays, kept = {}, defaultdict(list)
    for name in counts:
        for L, c in counts[name].items():
            if length_hist[name][L] < args.min_support:
                continue
            pssm = (c + args.pseudo) / (c.sum(1, keepdims=True) + 20 * args.pseudo)
            arrays[f"{name}__{L}"] = pssm.astype(np.float32)
            kept[name].append(L)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out + ".npz", **arrays)
    meta = {"aa_order": AA, "min_support": args.min_support, "n_seq": nseq,
            "length_hist": {k: dict(v) for k, v in length_hist.items()},
            "kept_lengths": {k: sorted(v) for k, v in kept.items()},
            # 방향족 통계 (길이별): 방향족 개수 분포 + W/Y/F 누적
            "arom_hist": {c: {L: dict(kk) for L, kk in d.items()} for c, d in arom_hist.items()},
            "wyf_sum": {c: {L: wyf[c][L].tolist() for L in wyf[c]} for c in wyf}}
    json.dump(meta, open(args.out + "_stats.json", "w"), indent=2)

    print(f"\n저장: {args.out}.npz ({len(arrays)} 길이별PSSM), {args.out}_stats.json")
    for name in ["H1", "H2", "H3", "L1", "L2", "L3"]:
        if name in length_hist:
            h = length_hist[name]
            tot = sum(h.values())
            mode = max(h, key=h.get)
            rng = f"{min(h)}~{max(h)}"
            print(f"  {name}: 길이범위 {rng}, 최빈 {mode} ({100*h[mode]/tot:.0f}%), "
                  f"PSSM 보유 길이 {sorted(kept[name])}")


if __name__ == "__main__":
    main()
