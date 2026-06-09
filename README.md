# esm_binder_design_base

논문 **Algorithm 11 (Gradient-Guided Binder Sequence Optimization)** 의 충실 재현 + **자연 CDR 분포 prior(composition)** 확장.
ESMFold2 의 미분 가능한 distogram 을 gradient 로 최적화해 항체 scFv 의 **CDR 만 재설계**(framework·항원 고정)하고,
ESMC LM prior + **composition(위치별 자연 분포)** 로 자연성을 규제, 저온에서 **real confidence ipTM** 으로 최적 설계를 선택,
**4-critic ipSAE 앙상블**로 랭킹한다.

## 핵심 결과 (composition 30+30 vs baseline)

| batch | best ipSAE | mean ipSAE | naturalness NLL | pLDDT | 다양성(identity) |
|---|---|---|---|---|---|
| baseline (no comp) | 0.145 | 0.038 | 4.95 | 74.4 | 0.33 |
| KL-comp | 0.440 | 0.174 | 0.82 | 81.9 | 0.86 |
| CE-comp | **0.488** | 0.154 | 0.89 | 81.2 | 0.84 |

→ **자연 CDR 분포 prior 가 4-critic 전이(ipSAE)를 best 3배·mean 4.6배 향상.** (설계 현실성↑ → 독립 critic 전이↑.)
새 과제: **다양성 붕괴**(germline 수렴). 상세 분석·figure: [`report/팀미팅_리포트.md`](report/팀미팅_리포트.md).

## 파이프라인

```
config -> run.py: scFv 조립 + 인덱스
  -> [생성] Algorithm 11: soft -> ESMFold2 distogram -> 구조손실(inter/intra/glob)
        + ESMC LM prior + composition(자연 CDR 분포) -> SGD -> b*
  -> [선택] 4-critic ipSAE 랭킹 -> 최종
```

## 폴더 구조

```
.
├── run.py                 # 드라이버 (생성 -> graft -> 랭킹)
├── optimize.py            # ★ Algorithm 11 코어 루프
├── losses.py              # 구조 손실 (inter/intra_contact, globularity, masked_pseudo_ppl)
├── composition.py         # ★ 자연 CDR 분포 prior (KL / CE), nat_nll
├── esmc_prior.py          # ESMC-600M masked pseudo-PPL (LM prior)
├── esmfold_diff.py        # 미분 distogram forward + 저온 real ipTM
├── model_hooks.py         # ESMFold2 설계용 배선 (confidence NoOp/real swap)
├── rank.py / rank_all.py  # 4-critic ipSAE 랭킹
├── scfv.py critics.py metrics.py run_esmfold2.py
├── full_run.sh            # 풀 실행 (GPU 병렬 생성 -> 통합 랭킹)
├── configs/               # 타깃 config (trastuzumab-HER2)
├── data/                  # PSSM 빌드 코드 + 소형 PSSM(.npz)
│   ├── build_pssm.py           # 위치별 q_target PSSM (설계 CDR 위치 매핑)
│   ├── build_length_pssm.py    # 길이층화 PSSM (CDR x length x position) + 방향족 통계
│   ├── download_oas_colab.py   # OAS paired 다운로드 (Colab -> Drive)
│   ├── trastuzumab_qtarget_oas.npz   # composition q_target (OAS+TheraSAbDab)
│   └── length_pssm.npz + _stats.json # 56개 길이별 PSSM + 길이/방향족 통계
├── report/                # ★ 데이터분석 (figure + 분석 스크립트 + 리포트)
│   ├── 팀미팅_리포트.md
│   ├── make_figures.py make_architecture.py make_cdr_compare.py
│   ├── make_length_figures.py analyze_batch.py
│   └── *.png  (fig0 아키텍처, fig1~6, fig4b, figA/B 비교, figL1~3 길이)
└── docs/                  # PLAN.md, 코드가이드.md, 베이스라인정리.md
```

## composition PSSM 데이터 소스 + 빌드

위치별 자연 분포 `q_target` 는 **자연 항체 서열**에서 ANARCI(IMGT) 넘버링으로 구축한다.

**소스:**
- **TheraSAbDab** (therapeutic 항체, 로컬 CSV) — `HeavySequence`/`LightSequence`.
- **OAS paired** (Observed Antibody Space, 60만 쌍) — Colab 에서 다운로드 후 `data/oas_paired_vdomains.csv` 로 복사
  (compute 노드 firewall → `data/download_oas_colab.py` 사용. 원천 CSV 는 대용량이라 레포 미포함).

**빌드 (별도 anarci env 필요, 예: `conda create -n abnum -c bioconda anarci`):**
```bash
# 위치별 q_target (composition 손실용)
python data/build_pssm.py --refs <TheraSAbDab.csv> data/oas_paired_vdomains.csv \
    --out data/trastuzumab_qtarget_oas.npz --ncpu 14
# 길이층화 PSSM (variable-length 설계용) + 길이/방향족 통계
python data/build_length_pssm.py --refs <TheraSAbDab.csv> data/oas_paired_vdomains.csv \
    --out data/length_pssm --ncpu 14
```
(소형 `.npz` 는 레포에 포함되어 있어 바로 사용 가능.)

## 실행

```bash
conda env create -f environment.yml && conda activate esmfold2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 단일 생성 + composition(CE) + 랭킹
CUDA_VISIBLE_DEVICES=0 python run.py --trajectories 2 --steps 150 \
    --pssm data/trastuzumab_qtarget_oas.npz --comp-loss ce --lambda-comp 0.5 \
    --rank --rank-msa auto --wandb --wandb-project ESMCRAFT

# 풀 실행 (GPU 병렬)
bash full_run.sh 2 150 6
```
토글: `--comp-loss {kl,ce}` · `--lambda-comp` · `--no-lm` · `--no-real-iptm` · `--wandb`

## 가중치 경로 (환경변수)

| 환경변수 | 기본 |
|---|---|
| `ESMFOLD2_WEIGHTS` | `/home/aidx/DB/weights/esmfold2/ESMFold2` |
| (ESMC-600M) | `/home/aidx/DB/weights/esmfold2/ESMC-600M` |
| (critics) | `/home/aidx/DB/weights/esmfold2/esmfold2_critics` |

> 가중치는 레포 미포함(경로 참조만). ESMFold2 는 커스텀 fork(`Biohub/transformers`, `Biohub/esm`) 의존 — `requirements.txt` 참조.
