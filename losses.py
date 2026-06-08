"""losses.py — 논문 Algorithm 11/12/13/14 손실 (깨끗한 재현).

모두 ESMFold2 trunk distogram 에서 (또는 ESMC LM 에서) 미분 가능하게 계산.
  · inter_contact   (Alg12) : binder-target cross-chain confident contact   → 낮을수록↓
  · intra_contact   (Alg12) : binder 내부 confident contact (de novo 접힘)   → 낮을수록↓
  · globularity     (Alg13) : binder 가 퍼지지 않고 globular                  → 낮을수록↓
  · masked_pseudo_ppl (Alg14): ESMC masked pseudo-perplexity (서열 자연성)    → 낮을수록↓
  · interface_tm    (참고)   : distogram 기반 인터페이스 신뢰도 proxy         → 높을수록↑
distogram bins: linspace(2.0, 22.0, 65) = 64 bins (ESMFold2 base structure_head).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

DISTOGRAM_MIN, DISTOGRAM_MAX, DISTOGRAM_BINS = 2.0, 22.0, 64


def bin_centers(device=None, dtype=torch.float32) -> torch.Tensor:
    e = torch.linspace(DISTOGRAM_MIN, DISTOGRAM_MAX, DISTOGRAM_BINS + 1, device=device, dtype=dtype)
    return 0.5 * (e[:-1] + e[1:])


def _con_map(distogram_logits, cutoff, binary=True):
    """-log P(dist<cutoff) per pair (colabdesign con-loss). 낮을수록 contact 강함."""
    c = bin_centers(distogram_logits.device)
    dg = distogram_logits.float()
    bins = (c < cutoff).float()
    px = F.softmax(dg, dim=-1)
    if binary:
        return -torch.log((bins * px).sum(-1) + 1e-8)
    px_ = F.softmax(dg - 1e7 * (1 - bins), dim=-1)
    return -(px_ * F.log_softmax(dg, dim=-1)).sum(-1)


def contact_probability(distogram_logits, cutoff=8.0):
    c = bin_centers(distogram_logits.device)
    p = F.softmax(distogram_logits.float(), dim=-1)
    return (p * (c < cutoff)).sum(-1)


# ── Alg 12: InterContactLoss (binder ↔ target) ────────────────────────────────
def inter_contact(distogram_logits, binder_idx, target_idx, cutoff=20.0, num=2):
    """각 binder 잔기가 target 잔기 중 가장 가까운 num개와 confident contact. 낮을수록↓."""
    x = _con_map(distogram_logits, cutoff)
    sub = x[..., binder_idx, :][..., :, target_idx]
    k = min(num, sub.shape[-1])
    return sub.topk(k, dim=-1, largest=False).values.mean()


# ── Alg 12: IntraContactLoss (binder 내부) ────────────────────────────────────
def intra_contact(distogram_logits, binder_idx, cutoff=14.0, num=2, seqsep=9):
    """binder 내부 (seqsep≥) confident contact → de novo 접힘. 낮을수록↓.
    유효(seqsep 통과) 쌍만 평균. scFv 처럼 binder 가 길면 의미 있음."""
    x = _con_map(distogram_logits, cutoff)
    idx = torch.as_tensor(binder_idx, device=distogram_logits.device)
    sub = x[..., idx, :][..., :, idx]
    sep = (idx[:, None] - idx[None, :]).abs() >= seqsep
    sub = sub.masked_fill(~sep, float("inf"))
    k = min(num, sub.shape[-1])
    topk = sub.topk(k, dim=-1, largest=False).values
    fin = torch.isfinite(topk)
    return topk[fin].mean() if bool(fin.any()) else distogram_logits.new_zeros(())


# ── Alg 13: GlobularityLoss ───────────────────────────────────────────────────
def globularity(distogram_logits, binder_idx):
    """binder 내부 기대거리 평균(Rg proxy), DISTOGRAM_MAX 로 정규화(0~1). 늘어짐 억제. 낮을수록↓.
    정규화 이유: 원거리(~18Å)는 contact 손실(-logP, ~0-3)과 스케일이 달라 L_struct 지배 →
    /DISTOGRAM_MAX 로 ~O(1) 맞춰 intra/inter 와 균형."""
    c = bin_centers(distogram_logits.device)
    idx = torch.as_tensor(binder_idx, device=distogram_logits.device)
    p = F.softmax(distogram_logits.float(), dim=-1)
    exp_d = (p * c).sum(-1)
    return exp_d[..., idx, :][..., :, idx].mean() / DISTOGRAM_MAX


# ── 참고: distogram 기반 인터페이스 신뢰도 proxy (ranking distogram-proxy 용) ──
def _d0(n):
    n = torch.clamp(n.float(), min=20.0)
    return 1.24 * (n - 15.0) ** (1.0 / 3.0) - 1.8


def interface_tm(distogram_logits, binder_idx, target_idx):
    """expected-distance 기반 distance-TM, epitope-restricted. 0..1, 높을수록↑."""
    c = bin_centers(distogram_logits.device)
    p = F.softmax(distogram_logits.float(), dim=-1)
    exp_d = (p * c).sum(-1)
    if exp_d.dim() == 3:
        exp_d = exp_d[0]
    L = exp_d.shape[-1]; dev = exp_d.device
    d0 = _d0(torch.tensor(float(len(binder_idx) + len(target_idx)), device=dev))
    tm = 1.0 / (1.0 + (exp_d / d0) ** 2)
    mb = torch.zeros(L, device=dev); mb[torch.as_tensor(binder_idx, device=dev)] = 1.0
    mt = torch.zeros(L, device=dev); mt[torch.as_tensor(target_idx, device=dev)] = 1.0
    inter = mb[:, None] * mt[None, :] + mt[:, None] * mb[None, :]
    per_row = (tm * inter).sum(1) / (inter.sum(1) + 1e-8)
    return per_row.max()


# ── Alg 14: MaskedPseudoPPL (ESMC LM prior) ───────────────────────────────────
def masked_pseudo_ppl(esmc_score_fn, soft_binder, mutable_mask,
                      M: int = 4, r_mask: float = 0.15, seed: int = 0):
    """ESMC masked pseudo-perplexity. 서열을 자연스럽게 유지(낮을수록↓).
    soft_binder (L,20) softmax 분포. mutable_mask (L,) bool. M회 마스킹 평균.
    esmc_score_fn(soft_binder, masked_positions) → masked 위치에서 -log P(soft) (미분가능).

    구현 메모: ESMC 는 ESMFold2 내부 LM. esmc_score_fn 은 esmfold_diff 에서 주입.
    (마스킹: mutable 위치에서 r_mask 비율을 매 pass 랜덤 선택; 6B 비용 고려해 M=4.)
    """
    if esmc_score_fn is None:
        return soft_binder.new_zeros(())
    g = torch.Generator(device="cpu").manual_seed(seed)
    mut = torch.where(mutable_mask)[0]
    if len(mut) == 0:
        return soft_binder.new_zeros(())
    n_mask = max(1, int(round(r_mask * len(mut))))
    total = soft_binder.new_zeros(())
    for m in range(M):
        perm = mut[torch.randperm(len(mut), generator=g)]
        masked = perm[:n_mask]
        total = total + esmc_score_fn(soft_binder, masked)
    return total / M
