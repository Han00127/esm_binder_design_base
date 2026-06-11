# ANARCI 넘버링 / 길이별 PSSM 빌드 — 환경·실행 가이드

> q_target.npz / length_pssm.npz 같은 **자연 CDR 분포(PSSM)** 를 만들 때 쓰는 ANARCI(IMGT 넘버링) 실행법.
> 2026-06-11 삽질 끝에 정리. **핵심 결론부터: 환경은 `abnum`, 반드시 inline(conda 살아있는 셸)으로 실행.**

## TL;DR (이대로만 하면 됨)

```bash
cd /home/kyeongtak/structure_projects/esm_binder_design_base
export PATH=/home/kyeongtak/.conda/envs/abnum/bin:$PATH
/home/kyeongtak/.conda/envs/abnum/bin/python -u data/build_length_pssm.py \
  --refs /home/aidx/DB/AGAB_MSADB/TheraSAbDab_SeqStruc_OnlineDownload.csv \
         data/oas_paired_vdomains.csv \
  --out data/length_pssm_full --max-refs 0 --ncpu 28 --min-support 30 \
  > runs/length_full_v2.log 2>&1
```
- `--max-refs 0` = subsample 없이 **전체** 사용 (속도 위해 줄이려면 양수 지정).
- 정상이면 로그에 `numbered N/602237 (기여 N)` 가 쌓이고 `hmmscan 에러: 0`.

## 환경 정리 (어느 env가 무엇)

| env | ANARCI python 모듈 | hmmscan | 비고 |
|---|---|---|---|
| **`abnum`** ✅ | **작동** (bioconda anarci) | `abnum/bin/hmmscan` | **PSSM 빌드는 여기서** |
| `esmfold2` | ❌ biopython 비호환 → `_domains_are_same`서 `TypeError(NoneType)` | `esmfold2/bin/hmmscan` | 메인 파이프라인용. raw `anarci` 깨짐 → abnum을 따로 만든 이유 |
| `mpnn` | (AntiFold pip 설치돼 있음) | — | 사용 안 함 |

- AntiFold(`/home/kyeongtak/structure_projects/inversefold/AntiFold`)는 esmfold2/mpnn에 pip 설치돼 있으나, 그 raw `anarci`는 위 biopython 이슈로 깨짐. **numbering은 abnum이 정답.**

## ★ 갑자기 "안 되는" 진짜 원인 (이걸로 한참 헤맴)

증상: `FileNotFoundError: [Errno 2] No such file or directory: 'hmmscan'` 가 모든 batch에서 → 결과 0개.

원인: **conda 런타임을 잃은 채 실행**. ANARCI는 내부적으로 `hmmscan` 바이너리를 subprocess로 호출하는데,
- `.sh` 스크립트로 만들어 `setsid`/`nohup` 으로 **분리(detach)** 실행하면, 그 프로세스가 conda 환경(PATH + 공유 lib 경로)을 제대로 못 물려받음.
- hmmscan은 컴파일된 바이너리라 lib을 못 찾으면 exec 자체가 실패 → 커널이 `No such file or directory` 를 던짐 (파일은 있는데도!).

해결: **inline 으로 실행** = conda가 살아있는 현재 셸에서 직접(또는 harness/`run_in_background`로). `export PATH=.../abnum/bin:$PATH` 만 해주면 hmmscan 찾고 정상 작동.

### 하지 말 것 / 할 것
- ❌ `cat > build.sh <<EOF ... EOF; setsid nohup ./build.sh &` (env 잃음)
- ✅ 현재 셸에서 `export PATH=.../abnum/bin:$PATH` 후 `python ... &` 또는 harness 백그라운드
- ✅ 검증: 띄운 뒤 `grep -c hmmscan <log>` 가 **0**, `pgrep -c hmmscan` 가 **>0** 이어야 진짜 작동중

## 데이터 소스 (참고)

| 소스 | 개수 | 경로 |
|---|---|---|
| OAS paired V-domain | 600,000 | `data/oas_paired_vdomains.csv` (col `vdomain_aa`, `locus` H/L) |
| TheraSAbDab | 1,133 항체 (≈2,237 V-domain) | `/home/aidx/DB/AGAB_MSADB/TheraSAbDab_SeqStruc_OnlineDownload.csv` |

- 빌드 스크립트가 `vdomain_aa`(OAS) vs `HeavySequence`/`LightSequence`(TheraSAbDab) 자동 감지.
- 필터: 70 ≤ len ≤ 200, 표준 20 AA만.
- H-CDR 위치는 VH 도메인에서만, L-CDR 위치는 VL 도메인에서만 기여 → 위치별 support ≈ 전체의 절반.

## 산출물 데이터셋 (2종 유지)

| 파일 | subsample | 용도 |
|---|---|---|
| `data/length_pssm.npz` | `--max-refs 120000` | 기존(빠른) 버전 |
| `data/length_pssm_full.npz` | `--max-refs 0` (전체 60.2만) | **full 버전** (희귀 길이 strata support↑) |
| `data/trastuzumab_qtarget_oas.npz` | `--max-refs 80000` | 위치별 q (트라스투주맙 CDR 58위치) |

빌드 스크립트: `data/build_pssm.py`(고정 CDR 위치 q), `data/build_length_pssm.py`(CDR×길이 strata).
