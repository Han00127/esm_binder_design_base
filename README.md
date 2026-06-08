# esm_binder_design_base

논문 **Algorithm 11 (Gradient-Guided Binder Sequence Optimization)** + 후처리 랭킹의 충실 재현.
ESMFold2(base)의 미분 가능한 distogram을 gradient로 최적화해 항체 scFv의 **CDR만 재설계**하고
(framework·항원 고정), ESMC LM prior로 자연성을 규제, 저온 구간에서 **real confidence head ipTM**으로
최적 설계를 선택한 뒤, **4-critic ipSAE 앙상블**로 랭킹한다. **foundry(기준) baseline.**

## 파이프라인

```
config(항원 + VH/VL + CDR ranges + epitope)
  → run.py: scFv 조립 + 복합체 features + 인덱스
  → [생성 ×N traj] optimize.py = Algorithm 11
       soft = softmax(x/T) → ESMFold2 distogram → 구조손실(inter/intra/glob)
       + ESMC-600M LM prior → gradient 정규화 → SGD
       + (저온 T<0.05) argmax → fold → real ipTM → b* 추적
       → 설계 CDR → graft → scFv 후보
  → [선택] rank_all.py: top-K → 4 critic 폴딩(msa=auto) → ipSAE 평균 → ranked.json
```

## 파일

| 파일 | 역할 |
|---|---|
| `run.py` | 드라이버 (입력→생성→graft→랭킹) |
| `optimize.py` | ★Algorithm 11 코어 루프 |
| `losses.py` | 구조 손실 (inter/intra_contact, globularity, masked_pseudo_ppl) |
| `esmc_prior.py` | ESMC-600M masked pseudo-PPL (LM prior) |
| `esmfold_diff.py` | 미분 distogram forward + 저온 real ipTM |
| `model_hooks.py` | ESMFold2 설계용 배선 (confidence NoOp/real swap) |
| `rank_all.py` / `rank.py` | 4-critic ipSAE 랭킹 |
| `scfv.py` / `critics.py` / `metrics.py` | scFv 조립 / critic 경로 / ipSAE |
| `run_esmfold2.py` | ESMFold2 폴딩 (MSA auto/none) |

## 설치

```bash
conda env create -f environment.yml   # python 3.12 + requirements.txt (커스텀 fork 포함)
conda activate esmfold2
```

> ESMFold2는 커스텀 fork(`Biohub/transformers`, `Biohub/esm`)에 의존 — `requirements.txt`에 포함됨.

## 가중치 경로 (환경변수로 덮어쓰기)

| 환경변수 | 기본 |
|---|---|
| `ESMFOLD2_WEIGHTS` | `/home/aidx/DB/weights/esmfold2/ESMFold2` |
| (ESMC-600M) | `/home/aidx/DB/weights/esmfold2/ESMC-600M` |
| (critics) | `/home/aidx/DB/weights/esmfold2/esmfold2_critics` |
| `MSA_CACHE_DIR` | `/home/kyeongtak/structure_projects/msa` |
