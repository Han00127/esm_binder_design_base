"""optimize.py — Algorithm 11 (Gradient-Guided Binder Sequence Optimization) 충실 재현.

입력: 준비된 features(target+binder 복합체) + index 집합.
출력: 최적화 trajectory 의 candidate 들 (b* = best ipTM + 최종 서열).

Alg11 정확히:
  init logits(fixed=10, mutable~N(0,1e-4), Cys=-1e6) → mask →
  for k=1..K: T_k cosine, α_k=α_max·T_k, soft=softmax(x/T_k),
    (T≥0.05 ? distogram-only : +confidence ipTM 추적),
    L_struct=λ_intra·intra+λ_inter·inter+λ_glob·glob, L_LM=maskedPPL,
    성분별 gradient 정규화 → g=ĝ_struct+λ_LM·ĝ_LM → x -= α_k·g
  return b* (best ipTM).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

import losses


@dataclass
class Alg11Params:
    K: int = 150
    alpha_max: float = 0.1
    T_min: float = 0.01
    M: int = 4                 # LM passes
    r_mask: float = 0.15
    lambda_LM: float = 0.05    # antibody=0.05, minibinder=0.15
    lambda_intra: float = 0.5
    lambda_inter: float = 0.5
    lambda_glob: float = 0.2
    lambda_comp: float = 0.0   # composition-KL(자연 CDR 분포) 가중 (0=끔)
    low_temp: float = 0.05     # T<low_temp → confidence/ipTM 추적 구간
    iptm_steps: int = 50       # 저온 confidence forward diffusion steps
    distogram_sampling: int = 1  # 고온 distogram forward sampler steps(최소)


def init_logits(L, mutable_idx, prompt_ids, cys_col, device):
    """Alg11 line1: fixed=prompt logit 10, mutable~N(0,1e-4), Cys=-1e6. + grad mask(line2)."""
    x = torch.zeros(L, 20, device=device)
    fixed = torch.ones(L, dtype=torch.bool, device=device)
    for p in mutable_idx:
        fixed[p] = False
    # fixed: prompt AA logit = 10
    for p in range(L):
        if fixed[p] and prompt_ids[p] is not None:
            x[p, prompt_ids[p]] = 10.0
    # mutable: N(0,1e-4), Cys=-1e6
    mut = ~fixed
    x[mut] = torch.randn(int(mut.sum()), 20, device=device) * 1e-2
    x[:, cys_col] = torch.where(mut, torch.full_like(x[:, cys_col], -1e6), x[:, cys_col])
    x.requires_grad_(True)
    # grad mask: mutable & non-Cys = 1
    m = torch.zeros(L, 20, device=device)
    m[mut] = 1.0
    m[:, cys_col] = 0.0
    return x, m, mut


def _norm_grad(g, mask, n_mut):
    """Alg11 line24-25: ĝ = sqrt(n_mut)·(g⊙m)/||g⊙m||_F."""
    gm = g * mask
    return (n_mut ** 0.5) * gm / (gm.norm() + 1e-8)


def optimize_binder(model, raw_fwd, feats, *, binder_idx, target_idx, mutable_idx,
                    prompt_ids, cys_col, build_soft_full, fold_idx=None, esmc_score_fn=None,
                    iptm_fn=None, comp_fn=None, metrics_cb=None, arom_cols=None, seed=0,
                    params: Alg11Params = Alg11Params(), device="cuda", log=print):
    """Algorithm 11 실행. 반환: dict(best_seq_logits, best_iptm, final_logits, history).

    build_soft_full(soft_binder_L20, T) → 모델 입력용 res_type 분포 (target onehot + binder soft 배치).
    esmc_score_fn(soft_binder, masked_pos) → masked pseudo-ppl 항 (없으면 LM prior 생략).
    iptm_fn(x_logits) → 저온 진짜 ipTM (real confidence head fold; 없으면 distogram proxy 폴백).
    seed → trajectory 별 mutable logits 초기화 다양화. b* = 저온 최고 ipTM 의 argmax 설계.
    """
    torch.manual_seed(seed)
    L = feats["res_type"].shape[1]
    x, gmask, mut = init_logits(L, mutable_idx, prompt_ids, cys_col, device)
    P = params
    n_mut = max(1, len(mutable_idx))
    best_iptm, best_logits = -1.0, x.detach().clone()
    hist = []
    t_loop = time.time()

    for k in range(1, P.K + 1):
        Tk = P.T_min + (1 - P.T_min) * 0.5 * (1 + math.cos(math.pi * k / P.K))  # line5
        ak = P.alpha_max * Tk                                                    # line6
        # ── 구조 손실 (ESMFold2 distogram) → g_struct, 그래프 *즉시 해제* ──
        soft_s = F.softmax(x / Tk, dim=-1)                                       # line7 (struct용)
        arom_frac = (float(soft_s[mutable_idx][:, arom_cols].sum(-1).mean())     # 진단: 방향족 분율
                     if arom_cols is not None else 0.0)
        soft_full = build_soft_full(soft_s, Tk)                                  # line9 concat
        dgram = esmfold_distogram(model, raw_fwd, feats, soft_full, P)           # line11 (grad)
        fold = fold_idx if fold_idx is not None else binder_idx
        L_intra = losses.intra_contact(dgram, fold)
        L_inter = losses.inter_contact(dgram, binder_idx, target_idx)
        L_glob = losses.globularity(dgram, fold)
        L_struct = P.lambda_intra * L_intra + P.lambda_inter * L_inter + P.lambda_glob * L_glob  # line21
        # distogram-ipTM proxy (b* 추적용; dgram 재사용 → 추가 forward 0). 높을수록↑
        disto_iptm = float(losses.interface_tm(dgram, binder_idx, target_idx))
        g_struct = torch.autograd.grad(L_struct, x, retain_graph=False)[0]       # line23 (ESMFold2 그래프 해제)
        del soft_full, dgram, soft_s

        # ── 저온(T<0.05): 진짜 ipTM 으로 b* 추적 (Alg11 line12-15) ──
        #   update *전* 현재 설계(=argmax(soft_k), 반환될 b*)를 real confidence head 로 평가.
        #   iptm_fn 없으면(=근사) distogram-ipTM proxy 로 폴백.
        if Tk < P.low_temp:
            if iptm_fn is not None:
                track_iptm = iptm_fn(x.detach())                                 # line13 real ipTM_k
            else:
                track_iptm = disto_iptm                                          # 근사 폴백
            if track_iptm > best_iptm:                                           # line14-15
                best_iptm, best_logits = track_iptm, x.detach().clone()

        # ── LM prior (ESMC) → g_LM, *별도 softmax*(구조 그래프 해제 후라 공존 X → OOM 방지) ──
        if esmc_score_fn is not None and P.lambda_LM > 0:
            soft_l = F.softmax(x / Tk, dim=-1)                                   # LM용 독립 그래프
            L_LM = losses.masked_pseudo_ppl(esmc_score_fn, soft_l, mut,
                                            M=P.M, r_mask=P.r_mask, seed=k)
            g_LM = (torch.autograd.grad(L_LM, x)[0]
                    if L_LM.requires_grad else torch.zeros_like(x))
        else:
            L_LM, g_LM = x.new_zeros(()), torch.zeros_like(x)

        # ── composition-KL (자연 CDR 분포) → g_comp, *별도 softmax* (cheap, 모델 forward 없음) ──
        if comp_fn is not None and P.lambda_comp > 0:
            soft_c = F.softmax(x / Tk, dim=-1)
            L_comp = comp_fn(soft_c[mutable_idx])                # [n_mut,20] (cdr_idx 순서)
            g_comp = (torch.autograd.grad(L_comp, x)[0]
                      if L_comp.requires_grad else torch.zeros_like(x))
        else:
            L_comp, g_comp = x.new_zeros(()), torch.zeros_like(x)

        # ── gradient 정규화 + 결합 + SGD (line24-27) ──
        g = (_norm_grad(g_struct, gmask, n_mut)
             + P.lambda_LM * _norm_grad(g_LM, gmask, n_mut)
             + P.lambda_comp * _norm_grad(g_comp, gmask, n_mut))
        with torch.no_grad():
            x -= ak * g
        if x.grad is not None:
            x.grad = None

        m = {"k": k, "T": Tk, "L_intra": float(L_intra), "L_inter": float(L_inter),
             "L_glob": float(L_glob), "L_LM": float(L_LM), "L_comp": float(L_comp),
             "arom_frac": arom_frac, "disto_iptm": disto_iptm, "best_iptm": best_iptm}
        hist.append(m)
        if metrics_cb is not None:
            metrics_cb(m)
        if k % 25 == 0 or k == P.K:
            log(f"  [k={k:3d}] T={Tk:.3f} inter={float(L_inter):.3f} intra={float(L_intra):.3f} "
                f"glob={float(L_glob):.3f} comp={float(L_comp):.3f} disto_iptm={disto_iptm:.3f}")

    dt = time.time() - t_loop
    log(f"[opt] {P.K} steps in {dt:.0f}s = {dt / P.K:.2f}s/step "
        f"(LM={'on' if esmc_score_fn else 'off'})")
    final_logits = x.detach().clone()
    if best_iptm < 0:                      # 저온 추적 없었으면 최종 서열
        best_logits = final_logits
    return {"best_logits": best_logits, "best_iptm": best_iptm,
            "final_logits": final_logits, "history": hist}


def esmfold_distogram(model, raw_fwd, feats, soft_full, P):
    """distogram_forward 래퍼 (esmfold_diff import 순환 피하려 여기서)."""
    from esmfold_diff import distogram_forward
    return distogram_forward(model, raw_fwd, feats, soft_full,
                             num_sampling_steps=P.distogram_sampling)
