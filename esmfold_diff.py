"""esmfold_diff.py — 미분가능 ESMFold2 forward (Algorithm 11 용).

두 모드:
  · distogram_forward : trunk → distogram_logits (grad O). 고온(T≥0.05) + 모든 step 의 손실용.
                        diffusion sampler 최소화(num_sampling_steps 작게) → 저렴.
  · iptm_confidence   : 저온(T<0.05) real confidence head swap-in → no_grad fold → 진짜 ipTM.
                        b* 추적용(Alg11 line12-15). argmax(soft) 이산 scFv 입력.

검증된 model_hooks(load_design_model, forward_design) 재활용. 생성 base = biohub/ESMFold2
(벤치마크에서 antibody-antigen 정확도 최고 확인).
"""
from __future__ import annotations

import os

import torch

from model_hooks import load_design_model, forward_design

DEFAULT_WEIGHTS = os.environ.get("ESMFOLD2_WEIGHTS", "/home/aidx/DB/weights/esmfold2/ESMFold2")


def load(weights: str = DEFAULT_WEIGHTS, device: str = "cuda"):
    """설계용 ESMFold2 로드 (distogram-only grad; confidence head 는 ipTM forward 에서 별도)."""
    model, raw_fwd = load_design_model(weights, device, enable_confidence_grad=False)
    return model, raw_fwd


def distogram_forward(model, raw_fwd, feats, soft, num_sampling_steps: int = 1):
    """soft 서열 → distogram_logits (grad). Alg11 line 11 (trunk+distogram).
    num_sampling_steps 작게 → sampler 최소(논문 'trunk and distogram only' 근사)."""
    out = forward_design(model, raw_fwd, feats, soft,
                         num_loops=1, num_sampling_steps=num_sampling_steps,
                         num_diffusion_samples=1, autocast=True)
    return out["distogram_logits"]


@torch.no_grad()
def iptm_confidence(model, builder, antigen_id, antigen_seq, binder_id, binder_seq,
                    num_sampling_steps: int = 50, seed: int = 0):
    """저온(T<0.05) 진짜 ipTM (Alg11 line12-15). 설계 모델의 real confidence head 를 일시
    복원(swap-in)해 no_grad full fold(diffusion steps + confidence) → result.iptm.

    입력 binder_seq 는 현재 설계의 argmax(이산 scFv). T<0.05 에서 softmax≈one-hot 이므로
    soft 입력과 사실상 동일하며, 논문 b*=argmax(soft) 라 '실제 반환할 설계'를 그대로 평가.
    반환: 항원↔binder 인터페이스 ipTM(float) — pair_chains_iptm 우선, 없으면 global iptm.
    """
    from esm.models.esmfold2 import ProteinInput, StructurePredictionInput

    real_head = getattr(model, "_real_confidence_head", None)
    if real_head is None:
        return float("nan")                       # confidence-intact 미보관 → 추적 불가
    noop_head = model.confidence_head
    model.confidence_head = real_head             # swap-in (real confidence)
    try:
        spi = StructurePredictionInput(sequences=[
            ProteinInput(id=antigen_id, sequence=antigen_seq),
            ProteinInput(id=binder_id, sequence=binder_seq)])      # MSA 없음(설계 forward 와 동일)
        raw = builder.fold(model, spi, num_loops=1, num_sampling_steps=num_sampling_steps,
                           num_diffusion_samples=1, seed=seed)
        result = raw[0] if isinstance(raw, list) else raw
    finally:
        model.confidence_head = noop_head         # swap-back (NoOp; 다음 step gradient 저렴)

    # 항원↔binder pair ipTM (2-chain: pair_chains_iptm[0,1]) 우선
    pcm = getattr(result, "pair_chains_iptm", None)
    if pcm is not None:
        try:
            m = pcm.float().cpu().numpy()
            if m.shape == (2, 2):
                return float(max(m[0, 1], m[1, 0]))
        except Exception:
            pass
    v = getattr(result, "iptm", None)             # fallback: global ipTM (2-chain≈인터페이스)
    try:
        return float(v) if v is not None else float("nan")
    except Exception:
        return float("nan")
