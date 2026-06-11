"""length_pssm 리포트: ① CDR별 길이 분포 ② CDR×길이별 PSSM(AA 빈도) 히트맵.
사용: python make_length_pssm_report.py <prefix>   (prefix=length_pssm | length_pssm_full)
"""
import json
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir("/home/kyeongtak/structure_projects/esm_binder_design_base")
PREFIX = sys.argv[1] if len(sys.argv) > 1 else "length_pssm_full"
npz = np.load(f"data/{PREFIX}.npz")
stats = json.load(open(f"data/{PREFIX}_stats.json"))
AA = stats["aa_order"]                       # 'ARNDCQEGHILKMFPSTWYV'
CDRS = ["H1", "H2", "H3", "L1", "L2", "L3"]
lh = stats["length_hist"]                    # CDR -> {length(str): count}
TAG = "FULL (전체 60만)" if "full" in PREFIX else "subsample"
print(f"=== length_pssm 리포트 [{PREFIX}]  n_seq={stats.get('n_seq')} ===")

# ── 콘솔: CDR별 길이 분포 요약 ──
for c in CDRS:
    d = {int(k): v for k, v in lh.get(c, {}).items()}
    tot = sum(d.values())
    mode = max(d, key=d.get) if d else None
    top = sorted(d.items(), key=lambda x: -x[1])[:5]
    print(f"  {c}: 총 {tot} | 최빈 길이 {mode} | top: " +
          " ".join(f"L{l}:{n}({n/tot*100:.0f}%)" for l, n in top))

# ── Figure 1: CDR별 길이 분포 ──
fig, axes = plt.subplots(2, 3, figsize=(15, 7))
for ax, c in zip(axes.flat, CDRS):
    d = {int(k): v for k, v in lh.get(c, {}).items()}
    if not d:
        ax.axis("off"); continue
    Ls = sorted(d)
    cnts = [d[l] for l in Ls]
    mode = max(d, key=d.get)
    cols = ["#e53935" if l == mode else "#1565c0" for l in Ls]
    ax.bar(Ls, cnts, color=cols)
    ax.set_title(f"{c}  (n={sum(cnts)}, mode=L{mode})", fontsize=11, weight="bold")
    ax.set_xlabel("CDR length"); ax.set_ylabel("count")
    ax.set_xticks(Ls)
fig.suptitle(f"CDR length distribution (IMGT) — {TAG}", fontsize=14, weight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f"report/fig_{PREFIX}_lengthdist.png", dpi=140, bbox_inches="tight")
print(f"saved: report/fig_{PREFIX}_lengthdist.png")

# ── Figure 2: CDR×길이별 PSSM 히트맵 (각 CDR 최빈 길이 top-3) ──
NTOP = 3
fig, axes = plt.subplots(len(CDRS), NTOP, figsize=(4.2 * NTOP, 3.0 * len(CDRS)))
for r, c in enumerate(CDRS):
    d = {int(k): v for k, v in lh.get(c, {}).items()}
    tops = [l for l, _ in sorted(d.items(), key=lambda x: -x[1])[:NTOP]]
    for j in range(NTOP):
        ax = axes[r, j]
        if j >= len(tops) or f"{c}__{tops[j]}" not in npz:
            ax.axis("off"); continue
        L = tops[j]
        pssm = npz[f"{c}__{L}"]              # [L, 20]
        im = ax.imshow(pssm.T, aspect="auto", cmap="viridis", vmin=0, vmax=max(0.4, pssm.max()))
        ax.set_yticks(range(20)); ax.set_yticklabels(list(AA), fontsize=6)
        ax.set_xticks(range(L)); ax.set_xticklabels(range(1, L + 1), fontsize=6)
        ax.set_title(f"{c}  len={L}  (n={d[L]})", fontsize=9, weight="bold")
        if j == 0:
            ax.set_ylabel("AA", fontsize=8)
        # 위치별 최빈 AA 표시
        for p in range(L):
            mi = pssm[p].argmax()
            ax.text(p, mi, AA[mi], ha="center", va="center", fontsize=6,
                    color="white" if pssm[p, mi] > 0.5 else "black")
fig.colorbar(im, ax=axes, shrink=0.4, label="frequency", pad=0.01)
fig.suptitle(f"CDR×length PSSM (IMGT, top-3 lengths) — {TAG}\n"
             "(글자 = 위치별 최빈 AA)", fontsize=14, weight="bold")
plt.savefig(f"report/fig_{PREFIX}_pssm.png", dpi=140, bbox_inches="tight")
print(f"saved: report/fig_{PREFIX}_pssm.png")
