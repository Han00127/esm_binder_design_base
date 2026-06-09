# esm_binder_design_base — 논문 Algorithm 11 충실 재현 (foundry baseline)

## 목적
ESMFold2 논문(Gradient-Guided Binder Sequence Optimization, Algorithm 11)의 설계 파이프라인을
**처음부터 충실히** 재현한다. esmcraft의 점진 개선이 baseline을 못 넘은 상황 →
**검증된 논문 레시피를 정확히 세우고**, 항원-항체(scFv)에서 실제 성능을 측정해 **foundry(기반)** 로 삼는다.
검증된 저수준 부품(미분가능 forward, distogram 수학)은 esmcraft에서 가져오되 **깨끗이 재구성**.

## 핵심 원리 (논문)
ESMFold2 = "한 번에 서열 생성"이 아니라 **미분가능 서열 평가자**.
- 미분 신호 = **distogram + confidence-derived output** (diffusion 좌표 경로 미경유 → 미분 쉬움).
- 대부분 step: `loops=1, trunk+distogram only` (저렴). 후반 저온: diffusion+confidence head로 ipTM 추적.
- 손실 3종(distogram) + ESMC LM prior. temperature annealing으로 soft→discrete 수렴.

## Algorithm 11 → 모듈 매핑

| Alg11 | 구현 |
|---|---|
| 1: logits 초기화 (fixed=10, mutable~N(0,1e-4), Cys=-1e6) | `optimize.init_logits` |
| 2: gradient mask (mutable & non-Cys) | `optimize` |
| 5: `T_k = T_min+(1-T_min)·½(1+cos(πk/K))` | `optimize` (K=150, T_min=0.01) |
| 6: `α_k = α_max·T_k` | `optimize` (α_max=0.1) |
| 7: `soft = softmax(x/T_k)` | `optimize` |
| 8: `F~Uniform{F1,F2}` (2 replicate) | `esmfold_diff` (모델 2개 랜덤; 메모리 = ESMC 공유 검토) |
| 9: `[onehot(target); soft_binder]` | `optimize` |
| 10-11: T≥0.05 → trunk+distogram only | `esmfold_diff.forward_distogram` |
| 12-15: else → +confidence, ipTM 추적 → b* | `esmfold_diff.forward_with_confidence` + `optimize` |
| 18: IntraContactLoss(D) | `losses.intra_contact` (Alg12) |
| 19: InterContactLoss(D) | `losses.inter_contact` (Alg12) |
| 20: GlobularityLoss(D) | `losses.globularity` (Alg13) |
| 21: `L_struct=λ_intra·intra+λ_inter·inter+λ_glob·glob` (0.5,0.5,0.2) | `optimize` |
| 22: MaskedPseudoPPL(ESMC, soft, μ; M=4, r=0.15) | `losses.masked_pseudo_ppl` (Alg14) ★신규 |
| 23-26: 성분별 gradient 정규화 후 결합 (λ_LM) | `optimize` |
| 27: `x = x - α_k·g` (SGD) | `optimize` |
| 29: return b* (best ipTM) | `optimize` |

LM weight λ_LM: **antibody 0.05**, minibinder 0.15. Cys는 mutable에서 배제.

## 후처리 (ranking)
- 많은 후보 생성 → **4-critic ensemble** (Experimental Fast×2 + full×2)로 재예측 → **average ipTM** 랭킹.
- (high-compute: +15 baseXX, ipTM-score와 distogram-proxy 각각 평균 후 결합.)
- `rank.py` — esmcraft `critics.py`/`metrics.py` 재활용.

## 모듈
```
esmfold_diff.py   미분가능 ESMFold2 (distogram / +confidence). esmcraft model_hooks 기반.
losses.py         intra/inter/globularity (distogram) + masked_pseudo_ppl (ESMC)
optimize.py       Algorithm 11 메인 루프
rank.py           4-critic ipTM ensemble
run.py            CLI: target+binder(scFv) → optimize → rank
configs/          파라미터 + target/binder spec
```

## 재활용(검증됨) vs 신규
- 재활용: 미분가능 forward(model_hooks), distogram contact 수학(losses), ipTM/ipSAE(metrics), scFv/CDR(scfv, ab_cdr_design_v1), critics.
- 신규: **MaskedPseudoPPL(ESMC)**, Algorithm 11 정확한 루프(temp/LR/gradnorm/SGD/ipTM-track), 2-replicate.

## 검증 계획
1. 논문 파라미터 그대로 (K=150 등) 1 trajectory smoke.
2. 항원-항체 타겟(예: trastuzumab-HER2 또는 벤치 타겟)으로 다수 후보 → ranking → ipTM 분포.
3. baseline(foundry) 성능 기록 → 이후 개선의 기준선.
