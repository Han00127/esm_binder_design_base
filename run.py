"""run.py — Algorithm 11 드라이버 (antibody scFv vs antigen).

config(YAML: 항원 + VH/VL + CDR ranges + epitope) → scFv+항원 복합체 features →
인덱스/prompt/build_soft_full 배선 → optimize_binder(Alg11) → 설계 CDR 출력.

체인 순서 = [항원(target), scFv(binder)]  (논문 concat [onehot(target); soft_binder]).
20 AA logits ↔ res_type id 2..21 (연속), Cys = id 6 (j=4).
첫 smoke: 구조손실만(esmc/iptm 미배선). 이후 ESMC LM prior + ipTM 추적 추가.
"""
from __future__ import annotations

import argparse
import os

import torch
import yaml

import scfv as scfvmod
from esm.models.esmfold2 import ESMFold2InputBuilder, ProteinInput, StructurePredictionInput
from esmc_prior import ESMCPrior
from esmfold_diff import iptm_confidence, load
from optimize import Alg11Params, optimize_binder

AA_BASE = 2          # res_type id of first AA (Ala); 20 AAs = ids 2..21
CYS_J = 6 - AA_BASE  # Cys j-index in 0..19  (=4)
LINKER = "GGGGSGGGGSGGGGS"


def _clean(s):
    return s.replace(" ", "").replace("\n", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/trastuzumab_her2.yaml")
    ap.add_argument("--steps", type=int, default=None, help="K override (smoke 시 작게)")
    ap.add_argument("--trajectories", type=int, default=1, help="생성 trajectory 수")
    ap.add_argument("--no-lm", action="store_true", help="ESMC LM prior 끄기 (구조손실만)")
    ap.add_argument("--no-real-iptm", action="store_true",
                    help="저온 b* 추적을 distogram proxy 로 (기본=real confidence ipTM, Alg11 충실)")
    ap.add_argument("--rank", action="store_true", help="후처리 4-critic ipSAE 랭킹 수행")
    ap.add_argument("--rank-msa", default="auto", choices=["auto", "none"], help="랭킹 폴딩 MSA")
    ap.add_argument("--rank-critics", default=None, help="critic key 콤마구분 (기본 4개)")
    ap.add_argument("--rank-gpu", type=int, default=1, help="랭킹 폴딩 GPU(물리번호)")
    ap.add_argument("--out", default=None, help="후보 JSON 저장 경로 (병렬 생성용)")
    ap.add_argument("--seed-base", type=int, default=0, help="trajectory seed 오프셋(GPU별 고유)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    cfg = yaml.safe_load(open(args.config))

    vh = _clean(cfg["antibody"]["heavy"]["vh_sequence"])
    vl = _clean(cfg["antibody"]["light"]["vl_sequence"])
    ag = cfg["antigen"]; ag_id, ag_seq = ag["id"], _clean(ag["sequence"])
    scfv = scfvmod.make_scfv(vh, vl, LINKER, "VH-VL")["seq"]

    # 체인: [항원(target), scFv(binder)]
    builder = ESMFold2InputBuilder()
    chains = [ProteinInput(id=ag_id, sequence=ag_seq), ProteinInput(id="S", sequence=scfv)]
    feats, _ = builder.prepare_input(
        StructurePredictionInput(sequences=chains), device=dev)
    rt = feats["res_type"]; L = rt.shape[1]
    Lag, Lsc = len(ag_seq), len(scfv)
    assert L == Lag + Lsc, f"L={L} != {Lag}+{Lsc}"

    # 인덱스 (항원 먼저: offset 0; scFv offset = Lag)
    epitope = [int(e) for e in cfg["epitope_residues"]]          # 항원 0-based
    target_idx = epitope                                         # inter-contact target
    sc_off = Lag
    voff = len(vh) + len(LINKER)                                 # scFv 내 VL 시작
    cdr_idx = []
    for _n, (s, e) in cfg["antibody"]["heavy"]["cdr_ranges"].items():
        cdr_idx += [sc_off + p for p in range(s, e)]
    for _n, (s, e) in cfg["antibody"]["light"]["cdr_ranges"].items():
        cdr_idx += [sc_off + voff + p for p in range(s, e)]
    cdr_idx = sorted(set(cdr_idx))
    fold_idx = list(range(sc_off, sc_off + Lsc))                 # scFv 전체 (intra/glob)
    mutable_idx = cdr_idx                                        # 설계 = CDR

    # prompt_ids: 각 위치의 20-AA j (res_type id - 2), 표준 AA 아니면 None
    rt0 = rt[0].tolist()
    prompt_ids = [(int(i) - AA_BASE) if (AA_BASE <= int(i) <= AA_BASE + 19) else None
                  for i in rt0]

    def build_soft_full(soft_binder, T):
        """soft_binder (L,20) → 모델 res_type 분포 (1,L,33). AA = cols 2..21."""
        d = torch.zeros(1, L, rt.shape[-1] if rt.dim() == 2 else 33, device=dev)
        # rt.shape[-1] (=33) 안전 처리
        d = torch.zeros(1, L, 33, device=dev)
        d[0, :, AA_BASE:AA_BASE + 20] = soft_binder
        return d

    print(f"[run] L={L} (ag {Lag} + scFv {Lsc}) | CDR(mutable) {len(cdr_idx)} | "
          f"epitope {len(epitope)} | fold(scFv) {len(fold_idx)}")

    # ── ESMC LM prior (논문 Alg14, soft 문맥) ── binder(scFv) 부분서열에 CDR mask 로 동작
    esmc_score_fn = None
    if not args.no_lm:
        print("[run] load ESMC-600M LM prior …")
        prior = ESMCPrior(device=dev)

        def esmc_score_fn(soft_full, masked_full):
            soft_scfv = soft_full[sc_off:sc_off + Lsc]                 # binder(scFv) 부분
            masked_within = [int(p) - sc_off for p in masked_full]     # CDR → scFv 내 인덱스
            return prior.score(soft_scfv, masked_within)

    print("[run] load model (base biohub/ESMFold2) …")
    model, raw_fwd = load(device=dev)

    P = Alg11Params(lambda_LM=0.05)
    if args.steps:
        P.K = args.steps
    print(f"[run] Algorithm 11: K={P.K} α_max={P.alpha_max} T_min={P.T_min} "
          f"λ(intra,inter,glob,LM)=({P.lambda_intra},{P.lambda_inter},{P.lambda_glob},{P.lambda_LM})")

    # ── 다중 trajectory 생성 (Alg11) → 후보 (graft 설계CDR → scFv) ──
    from esm.models.esmfold2.constants import PROTEIN_3TO1, PROTEIN_RESIDUE_TO_RES_TYPE
    id2aa = {PROTEIN_RESIDUE_TO_RES_TYPE[k]: PROTEIN_3TO1[k] for k in PROTEIN_3TO1}

    # ── 저온 real ipTM b* 추적 (Alg11 line12-15): 현재 설계 argmax → 이산 scFv → real
    #    confidence head fold(no_grad) → 항원↔scFv ipTM. (--no-real-iptm 이면 None → proxy 폴백) ──
    iptm_fn = None
    if not args.no_real_iptm:
        print(f"[run] 저온 b* = real confidence ipTM (fold steps={P.iptm_steps}, MSA 없음)")

        def iptm_fn(xb):
            scfv_des = list(scfv)
            for p in cdr_idx:
                scfv_des[p - sc_off] = id2aa.get(int(xb[p].argmax()) + AA_BASE, "A")
            return iptm_confidence(model, builder, ag_id, ag_seq, "S", "".join(scfv_des),
                                   num_sampling_steps=P.iptm_steps, seed=0)
    else:
        print("[run] 저온 b* = distogram-ipTM proxy (--no-real-iptm)")

    candidates = []
    for t in range(args.trajectories):
        sd = args.seed_base + t
        print(f"\n[run] === trajectory {t+1}/{args.trajectories} (seed={sd}) ===")
        res = optimize_binder(
            model, raw_fwd, feats,
            binder_idx=cdr_idx, target_idx=target_idx, mutable_idx=mutable_idx,
            fold_idx=fold_idx, prompt_ids=prompt_ids, cys_col=CYS_J,
            build_soft_full=build_soft_full, esmc_score_fn=esmc_score_fn,
            iptm_fn=iptm_fn, seed=sd, params=P, device=dev)
        xb = res["best_logits"]
        scfv_des = list(scfv)
        for p in cdr_idx:
            scfv_des[p - sc_off] = id2aa.get(int(xb[p].argmax()) + AA_BASE, "A")
        cdrseq = "".join(id2aa.get(int(xb[p].argmax()) + AA_BASE, "X") for p in cdr_idx)
        candidates.append({"name": f"s{sd}", "scfv": "".join(scfv_des),
                           "disto_iptm": round(res["best_iptm"], 4), "cdr": cdrseq})
        print(f"  disto_iptm={res['best_iptm']:.4f} CDR={cdrseq}")

    del model, raw_fwd            # 설계 모델 해제 (랭킹 critic 메모리 확보)
    torch.cuda.empty_cache()

    if args.out:                  # 병렬 생성: 후보 JSON 저장 (랭킹은 통합 단계에서)
        import json
        json.dump({"antigen_id": ag_id, "antigen_seq": ag_seq, "candidates": candidates},
                  open(args.out, "w"))
        print(f"[run] 후보 {len(candidates)}개 저장 → {args.out}")

    # ── 후처리 4-critic ipSAE 랭킹 ──
    if args.rank:
        import rank as ranker
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "rank_out")
        ck = args.rank_critics.split(",") if args.rank_critics else None
        print(f"\n[run] === 4-critic ranking (msa={args.rank_msa}, GPU{args.rank_gpu}) ===")
        ranked = ranker.rank(candidates, ag_seq, ag_id, out_dir,
                             critic_keys=ck, msa=args.rank_msa, gpu=args.rank_gpu)
        print("\n[run] ★ RANKED (avg_ipsae 내림차순):")
        for r in ranked:
            print(f"  {r['name']}: avg_ipsae={r['avg_ipsae']} disto_iptm={r['disto_iptm']} CDR={r['cdr']}")
    else:
        print("\n[run] 후보:")
        for c in candidates:
            print(f"  {c['name']}: disto_iptm={c['disto_iptm']} CDR={c['cdr']}")


if __name__ == "__main__":
    main()
