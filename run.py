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

os.environ.setdefault("HF_HUB_OFFLINE", "1")        # 오프라인 클러스터: HF 네트워크 HEAD 호출 차단
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

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
    ap.add_argument("--pssm", default=None,
                    help="composition-KL 타깃 q_target.npz (지정 시 자연 CDR 분포 손실 켜짐)")
    ap.add_argument("--lambda-comp", type=float, default=0.5, help="composition 손실 가중치")
    ap.add_argument("--comp-loss", default="kl", choices=["kl", "ce"],
                    help="composition 손실: kl(분포매칭) | ce(프로파일 NLL, 폭발 회피)")
    ap.add_argument("--wandb", action="store_true", help="Weights & Biases 로깅 켜기")
    ap.add_argument("--wandb-project", default="esm_binder_design", help="wandb 프로젝트명")
    ap.add_argument("--wandb-name", default=None, help="wandb run 이름")
    ap.add_argument("--rank", action="store_true", help="후처리 4-critic ipSAE 랭킹 수행")
    ap.add_argument("--rank-msa", default="auto", choices=["auto", "none"], help="랭킹 폴딩 MSA")
    ap.add_argument("--rank-critics", default=None, help="critic key 콤마구분 (기본 4개)")
    ap.add_argument("--rank-gpu", type=int, default=1, help="랭킹 폴딩 GPU(물리번호)")
    ap.add_argument("--out", default=None, help="후보 JSON 저장 경로 (병렬 생성용)")
    ap.add_argument("--seed-base", type=int, default=0, help="trajectory seed 오프셋(GPU별 고유)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pipeline-gpus", default=None,
                    help="트렁크 블록을 여러 GPU 에 분산(pipeline 병렬). 예: '0,1,2' "
                         "(큰 L backward 가 1 GPU 천장(L≈850)을 넘을 때; CUDA_VISIBLE_DEVICES 가시 인덱스)")
    ap.add_argument("--alpha-max", type=float, default=None,
                    help="α_max override (Alg11 기본 0.1; K 축소 시 total path length 보정용)")
    ap.add_argument("--lambda-lm", type=float, default=0.05,
                    help="LM prior 가중치 λ_LM (기본 0.05; 줄이면 LM 영향↓)")
    args = ap.parse_args()
    dev = args.device
    cfg = yaml.safe_load(open(args.config))

    vh = _clean(cfg["antibody"]["heavy"]["vh_sequence"])
    vl = _clean(cfg["antibody"]["light"]["vl_sequence"])
    scfv = scfvmod.make_scfv(vh, vl, LINKER, "VH-VL")["seq"]

    # ── 항원 파싱: 단일(antigen.id/sequence + epitope_residues) | 다중체인(chains[].epitope) ──
    #    다중체인이라도 loss 는 원래대로(inter_contact). target_idx = 모든 체인 epitope 의 전역 concat 인덱스
    #    → binder 가 3 체인 epitope 전체로 당겨짐(원래 inter loss 그대로, quaternary 모드 아님).
    if "chains" in cfg:                                  # 다중항원
        ag_chains, epitope, _off = [], [], 0
        for c in cfg["chains"]:
            cid, cseq = c["id"], _clean(c["sequence"])
            ag_chains.append((cid, cseq))
            epitope += [_off + int(e) for e in c.get("epitope", [])]   # 체인-로컬 0-based → 전역
            _off += len(cseq)
    else:                                                # 단일항원 (기존 동작 유지)
        ag = cfg["antigen"]
        ag_chains = [(ag["id"], _clean(ag["sequence"]))]
        epitope = [int(e) for e in cfg["epitope_residues"]]
    ag_seq = "".join(s for _, s in ag_chains)            # concat 항원 (downstream 폴백용)
    ag_id = ag_chains[0][0]
    Lag = len(ag_seq)

    # 체인: [항원체인들(target), scFv(binder)]
    builder = ESMFold2InputBuilder()
    chains = [ProteinInput(id=cid, sequence=s) for cid, s in ag_chains]
    chains.append(ProteinInput(id="S", sequence=scfv))
    feats, _ = builder.prepare_input(
        StructurePredictionInput(sequences=chains), device=dev)
    rt = feats["res_type"]; L = rt.shape[1]
    Lsc = len(scfv)
    assert L == Lag + Lsc, f"L={L} != {Lag}+{Lsc}"

    target_idx = epitope                                 # inter-contact target (전역 concat 인덱스)
    sc_off = Lag
    voff = len(vh) + len(LINKER)                                 # scFv 내 VL 시작
    h_fix = cfg["antibody"]["heavy"].get("cdr_fix_prefix", {}) or {}   # CDR 앞 N개 native 고정
    l_fix = cfg["antibody"]["light"].get("cdr_fix_prefix", {}) or {}
    cdr_idx, mutable_idx = [], []
    for _n, (s, e) in cfg["antibody"]["heavy"]["cdr_ranges"].items():
        pos = [sc_off + p for p in range(s, e)]
        cdr_idx += pos
        mutable_idx += pos[int(h_fix.get(_n, 0)):]               # 앞 N개 제외 → 나머지만 설계
    for _n, (s, e) in cfg["antibody"]["light"]["cdr_ranges"].items():
        pos = [sc_off + voff + p for p in range(s, e)]
        cdr_idx += pos
        mutable_idx += pos[int(l_fix.get(_n, 0)):]
    cdr_idx = sorted(set(cdr_idx))                               # binder/loss = 전체 CDR
    mutable_idx = sorted(set(mutable_idx))                       # 설계 = CDR − 고정prefix
    fold_idx = list(range(sc_off, sc_off + Lsc))                 # scFv 전체 (intra/glob)

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

    print(f"[run] L={L} (ag {Lag} + scFv {Lsc}) | CDR {len(cdr_idx)} / mutable {len(mutable_idx)} "
          f"(native 고정 {len(cdr_idx) - len(mutable_idx)}) | epitope {len(epitope)} | fold(scFv) {len(fold_idx)}")

    # ── ESMC LM prior (논문 Alg14, soft 문맥) ── binder(scFv) 부분서열에 CDR mask 로 동작
    esmc_score_fn = None
    if not args.no_lm:
        print("[run] load ESMC-600M LM prior …")
        prior = ESMCPrior(device=dev)

        def esmc_score_fn(soft_full, masked_full):
            soft_scfv = soft_full[sc_off:sc_off + Lsc]                 # binder(scFv) 부분
            masked_within = [int(p) - sc_off for p in masked_full]     # CDR → scFv 내 인덱스
            return prior.score(soft_scfv, masked_within)

    # ── composition-KL (자연 CDR 분포 타깃, build_pssm.py 산출 npz) ──
    comp_fn, comp_target = None, None
    if args.pssm:
        from composition import CompositionTarget
        print(f"[run] load composition target: {args.pssm} (λ_comp={args.lambda_comp})")
        comp_target = CompositionTarget(args.pssm, device=dev)
        print(f"[run] q_target {comp_target.n} 위치, q_global 방향족(W+Y+F)="
              f"{float(comp_target.q_global[comp_target._arom_j].sum()):.3f}")

        _cf = comp_target.nll if args.comp_loss == "ce" else comp_target.kl

        def comp_fn(soft_cdr):                # soft_cdr [n_mut,20] (cdr_idx 순서)
            return _cf(soft_cdr)              # --comp-loss 로 kl/ce 선택

    if args.pipeline_gpus:
        os.environ["PIPELINE_GPUS"] = args.pipeline_gpus       # esmfold_diff.load 에서 분산 활성
        print(f"[run] pipeline 병렬 요청: GPUs {args.pipeline_gpus}")
    print("[run] load model (base biohub/ESMFold2) …")
    model, raw_fwd = load(device=dev)

    P = Alg11Params(lambda_LM=args.lambda_lm, lambda_comp=(args.lambda_comp if args.pssm else 0.0))
    if args.steps:
        P.K = args.steps
    if args.alpha_max is not None:
        P.alpha_max = args.alpha_max
    print(f"[run] Algorithm 11: K={P.K} α_max={P.alpha_max} T_min={P.T_min} "
          f"λ(intra,inter,glob,LM)=({P.lambda_intra},{P.lambda_inter},{P.lambda_glob},{P.lambda_LM})")

    # ── 다중 trajectory 생성 (Alg11) → 후보 (graft 설계CDR → scFv) ──
    from esm.models.esmfold2.constants import PROTEIN_3TO1, PROTEIN_RESIDUE_TO_RES_TYPE
    id2aa = {PROTEIN_RESIDUE_TO_RES_TYPE[k]: PROTEIN_3TO1[k] for k in PROTEIN_3TO1}
    arom_cols = [j for j in range(20) if id2aa.get(j + AA_BASE) in ("W", "Y", "F")]

    # ── Weights & Biases (config + per-step loss 로깅) ──
    wb, _gstep = None, [0]
    if args.wandb:
        import wandb
        wb = wandb.init(project=args.wandb_project, name=args.wandb_name, config={
            "K": P.K, "alpha_max": P.alpha_max, "T_min": P.T_min, "low_temp": P.low_temp,
            "lambda_intra": P.lambda_intra, "lambda_inter": P.lambda_inter,
            "lambda_glob": P.lambda_glob, "lambda_LM": P.lambda_LM, "lambda_comp": P.lambda_comp,
            "iptm_steps": P.iptm_steps, "trajectories": args.trajectories,
            "lm": (not args.no_lm), "real_iptm": (not args.no_real_iptm),
            "pssm": args.pssm, "comp_loss": args.comp_loss, "config": args.config,
            "antigen": ag_id, "L": L, "n_cdr": len(cdr_idx), "n_epitope": len(epitope)})
        print(f"[run] wandb: {wb.url}")

    # ── 저온 real ipTM b* 추적 (Alg11 line12-15): 현재 설계 argmax → 이산 scFv → real
    #    confidence head fold(no_grad) → 항원↔scFv ipTM. (--no-real-iptm 이면 None → proxy 폴백) ──
    iptm_fn = None
    if not args.no_real_iptm and len(ag_chains) == 1:
        print(f"[run] 저온 b* = real confidence ipTM (fold steps={P.iptm_steps}, MSA 없음)")

        def iptm_fn(xb):
            scfv_des = list(scfv)
            for p in cdr_idx:
                scfv_des[p - sc_off] = id2aa.get(int(xb[p].argmax()) + AA_BASE, "A")
            return iptm_confidence(model, builder, ag_id, ag_seq, "S", "".join(scfv_des),
                                   num_sampling_steps=P.iptm_steps, seed=0)
    elif len(ag_chains) > 1:
        print("[run] 저온 b* = distogram-ipTM proxy (다중항원 → real ipTM 폴드 미지원)")
    else:
        print("[run] 저온 b* = distogram-ipTM proxy (--no-real-iptm)")

    AROM = set("WYF")
    candidates = []
    for t in range(args.trajectories):
        sd = args.seed_base + t
        print(f"\n[run] === trajectory {t+1}/{args.trajectories} (seed={sd}) ===")

        def mcb(m, t=t):              # per-step wandb 로깅 (global step 연속)
            if wb is not None:
                wb.log({**m, "traj": t}, step=_gstep[0]); _gstep[0] += 1

        res = optimize_binder(
            model, raw_fwd, feats,
            binder_idx=cdr_idx, target_idx=target_idx, mutable_idx=mutable_idx,
            fold_idx=fold_idx, prompt_ids=prompt_ids, cys_col=CYS_J,
            build_soft_full=build_soft_full, esmc_score_fn=esmc_score_fn,
            iptm_fn=iptm_fn, comp_fn=comp_fn, metrics_cb=(mcb if wb else None),
            arom_cols=arom_cols, seed=sd, params=P, device=dev)
        xb = res["best_logits"]
        scfv_des = list(scfv)
        for p in cdr_idx:
            scfv_des[p - sc_off] = id2aa.get(int(xb[p].argmax()) + AA_BASE, "A")
        cdrseq = "".join(id2aa.get(int(xb[p].argmax()) + AA_BASE, "X") for p in cdr_idx)
        arom = sum(c in AROM for c in cdrseq) / max(1, len(cdrseq))     # 최종 CDR 방향족 분율
        # B* 자연성: 위치별 NLL (낮을수록 자연 분포에 가까움) + native 기준선
        nat_nll = native_nll = None
        if comp_target is not None:
            best_j = [int(xb[p].argmax()) for p in mutable_idx]   # comp 타깃 = mutable 위치 정렬
            nat_nll = round(comp_target.seq_nll(best_j), 3)
            nj = [prompt_ids[p] for p in mutable_idx]
            if all(j is not None for j in nj):
                native_nll = round(comp_target.seq_nll(nj), 3)
        candidates.append({"name": f"s{sd}", "scfv": "".join(scfv_des),
                           "disto_iptm": round(res["best_iptm"], 4), "cdr": cdrseq,
                           "arom": round(arom, 3), "nat_nll": nat_nll})
        print(f"  disto_iptm={res['best_iptm']:.4f} arom={arom:.2f} "
              f"nat_nll={nat_nll} (native={native_nll}) CDR={cdrseq}")
        if wb is not None:
            wb.log({"traj_best_iptm": res["best_iptm"], "traj_arom": arom,
                    "traj_nat_nll": nat_nll, "native_nat_nll": native_nll, "traj": t},
                   step=_gstep[0])
        if args.out:                  # 점진 저장: trajectory마다 갱신 → 중단해도 완료분 보존
            import json as _json
            _json.dump({"antigen_id": ag_id, "antigen_seq": ag_seq, "candidates": candidates},
                       open(args.out, "w"))
            print(f"  [save] {len(candidates)}/{args.trajectories} 후보 → {args.out}")

    del model, raw_fwd            # 설계 모델 해제 (랭킹 critic 메모리 확보)
    torch.cuda.empty_cache()

    import json
    if args.out:                  # 후보 JSON 저장
        json.dump({"antigen_id": ag_id, "antigen_seq": ag_seq, "candidates": candidates},
                  open(args.out, "w"))
        print(f"[run] 후보 {len(candidates)}개 저장 → {args.out}")

    # ── 후처리 4-critic ipSAE 랭킹 ──
    ranked = None
    if args.rank:
        import rank as ranker
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "rank_out")
        ck = args.rank_critics.split(",") if args.rank_critics else None
        print(f"\n[run] === 4-critic ranking (msa={args.rank_msa}, GPU{args.rank_gpu}) ===")
        ranked = ranker.rank(candidates, ag_seq, ag_id, out_dir,
                             critic_keys=ck, msa=args.rank_msa, gpu=args.rank_gpu)
        json.dump(ranked, open(os.path.join(out_dir, "ranked.json"), "w"), indent=2)
        print(f"\n[run] ★ RANKED (avg_ipsae 내림차순) → {out_dir}/ranked.json:")
        for r in ranked:
            print(f"  {r['name']}: avg_ipsae={r['avg_ipsae']} disto_iptm={r['disto_iptm']} "
                  f"arom={r.get('arom')} nat_nll={r.get('nat_nll')} CDR={r['cdr']}")
    else:
        print("\n[run] 후보:")
        for c in candidates:
            print(f"  {c['name']}: disto_iptm={c['disto_iptm']} arom={c.get('arom')} "
                  f"nat_nll={c.get('nat_nll')} CDR={c['cdr']}")

    # ── wandb 결과 Table (모든 설계 비교: disto_iptm / arom / nat_nll / avg_ipsae / CDR) ──
    if wb is not None:
        import wandb
        rows = ranked if ranked is not None else candidates
        cols = ["name", "disto_iptm", "arom", "nat_nll", "avg_ipsae", "cdr"]
        tbl = wandb.Table(columns=cols)
        for r in rows:
            tbl.add_data(r.get("name"), r.get("disto_iptm"), r.get("arom"),
                         r.get("nat_nll"), r.get("avg_ipsae"), r.get("cdr"))
        wb.log({"results": tbl})
        wb.summary["n_designs"] = len(candidates)
        wb.finish()


if __name__ == "__main__":
    main()
