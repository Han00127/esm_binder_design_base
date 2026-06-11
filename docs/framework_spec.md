# Framework Spec — Trastuzumab scaffold (설계 substrate 특성)

> 우리 파이프라인이 CDR을 graft하는 **고정 framework**의 특성 한 장 요약. (ANARCI/abnum 분석 + 구조 1N8Z + canonical/Vernier 문헌)
> 목적: 다양성 회복·loss-rebalance·길이 sampling을 "어느 CDR에 얼마나"라는 **구체 숫자**로 떨어뜨리는 근거.

## 1. 정체성 & germline (ANARCI assign_germline)

| 도메인 | V gene | J gene | family | 비고 |
|---|---|---|---|---|
| **VH** | human **IGHV3-66\*01** (id 0.82) | IGHJ4\*01 (0.93) | **VH3** | consensus 인간 VH3 |
| **VL** | human **IGKV1-39\*01** (id 0.87) | IGKJ1\*01 (0.92) | **Vκ1** (kappa) | consensus 인간 Vκ1 |

→ 전형적 **humanized consensus framework**(4D5/trastuzumab). VH3+Vκ1은 인간에서 가장 흔하고 canonical 구조가 잘 정의됨 → 일반화·humanness 유리.

## 2. FR/CDR 경계 + 길이 (우리 설계영역 = Kabat)

| CDR | 설계영역(서열 0-based) | Kabat 위치 | 서열 | 길이 |
|---  |-----------------------|---|---|---|
| H1  | 27–35  | 28–35         | NIKDTYIH | 8 |
| H2  | 50–62  | 51–61(+52A)   | IYPTNGYTRYAD | 12 |
| H3  | 98–109 | 95–102       | WGGDGFYAMDY | 11 |
| L1  | 23–34  | 24–34         | RASQDVNTAVA | 11 |
| L2  | 49–56  | 50–56 | SASFLYS | 7 |
| L3  | 88–97 | 89–97 | QQHYTTPPT | 9 |

## 3. CDR-anchoring 잔기 (framework가 CDR을 잡는 곳)

**Vernier zone** (CDR 밑을 받쳐 conformation 미세조정; Kabat, Foote&Winter):
- VH: `2V 27F 28N 29I 30K 47W 48V 49A 67F 69I 71A 73T 78A 93S 94R`
- VL: `2I 4M 35W 36Y 46L 47L 48I 49Y 64G 66R 68G 69T 71F 98F`

**VH/VL interface** (paratope 방향 결정 packing 잔기):
- VH: `37V 39Q 45L 47W 91Y 103W` · VL: `36Y 38Q 43A 44P 46L 87Y 98F`

### ★ 핵심 발견 — H1 설계영역이 Vernier와 겹침
```
H1 설계영역:  N28[V] I29[V] K30[V] D31 T32 Y33 I34 H35
                └─ Vernier(framework) ─┘  └── 진짜 CDR-H1 ──┘
```
- **H1 첫 3잔기(N28·I29·K30)는 Vernier zone framework** — CDR 자체가 아니라 H1·H2 loop를 *받치는* 잔기. 우리가 이걸 재설계 중.
- 나머지 CDR(H2/H3/L1/L2/L3 설계영역)은 Vernier/interface와 **겹침 없음**(깨끗).
- 구조적 증거와 일치: H1의 Cα RMSD(1.23Å)가 다른 보존 CDR(0.4–0.6Å)보다 큼 — Vernier 겹침 재설계 때문일 가능성.

## 4. Canonical class & graftability (germline+길이 기반)

| CDR | canonical | 제약 정도 | graft 가능 길이 |
|---|---|---|---|
| L1 | Vκ1 L1 (len 11) 표준 | **강** | canonical 호환 길이만(주로 11; Vκ1는 16도) |
| L2 | 사실상 단일 class(불변 backbone) | **매우 강** | len 7 고정적 |
| L3 | len 9, **Pro-kinked** | 강 | 8–10, cis-Pro 패턴 유지 |
| H1 | germline H1 canonical | 중 | Vernier(28–30) 보존 시 안정 |
| H2 | germline H2 canonical | 중 | len 호환 범위 |
| **H3** | **non-canonical** (kink rule만) | **약(자유)** | 광범위 — paratope 핵심 |

> 정밀 라벨(North/PyIgClassify)은 SAbDab로 확정 가능. 위는 germline+길이+1N8Z 구조 기반.

## 5. 설계 함의 (downstream 직결)

- **H3 = 자유 + paratope 지배** → **다양성·설계 자유도를 여기 집중.** (구조분석서 H3만 3.8Å 발산한 것과 일치)
- **L1/L2/L3·H2 = canonical 강제약** → 보존적으로, **길이는 canonical 호환 범위 내에서만** sampling (length PSSM과 결합).
- **H1 = Vernier 겹침 주의** → 첫 3위치(N28/I29/K30)는 **구속하거나 germline 유지** 권장(canonical 안정성·humanness 보호).
- **per-CDR λ 제안**: H3 약하게(자유 탐색) · L-CDR/H2 중간(canonical 유지) · **H1 Vernier 위치는 강하게 구속**. → loss-rebalance의 출발점.
- **항체 LM prior**: VH3/Vκ1 **germline-conditioned** prior가 pooled PSSM보다 정확(같은 germline 문맥).
- **일반화 검증**: 다른 germline framework(예: VH1, Vλ)로 2차 검증 시 위 표가 framework마다 재산출되어야 함 → 파이프라인이 framework를 입력으로 받는 구조라 가능.

---
*생성: ANARCI(abnum, IMGT+Kabat+germline) + 1N8Z 구조 + canonical/Vernier 문헌. 재현: `docs/ANARCI_PSSM_BUILD.md` 환경.*
