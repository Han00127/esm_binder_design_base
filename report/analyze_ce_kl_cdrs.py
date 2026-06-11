"""CE/KL 생성 CDR 전수 분석: region별 native-identity, L1 canonical 보존, H1 Vernier(28-30) 분석."""
import glob
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir("/home/kyeongtak/structure_projects/esm_binder_design_base")
plt.rcParams["axes.unicode_minus"] = False

NATIVE = "NIKDTYIHIYPTNGYTRYADWGGDGFYAMDYRASQDVNTAVASASFLYSQQHYTTPPT"
segs = [("H1", 8), ("H2", 12), ("H3", 11), ("L1", 11), ("L2", 7), ("L3", 9)]
bounds, off = {}, 0
for n, L in segs:
    bounds[n] = (off, off + L); off += L


def split(cdr):
    return {n: cdr[a:b] for n, (a, b) in bounds.items()}


def load(pats):
    out = []
    for pat in pats:
        for f in sorted(glob.glob(pat)):
            out += [c["cdr"] for c in json.load(open(f)).get("candidates", [])]
    return out


SETS = {"CE": load(["runs/batch_ce_g1.json", "runs/batch_ce_g2.json"]),
        "KL": load(["runs/batch30_kl.json"])}
nat = split(NATIVE)

print("=" * 72)
print(" CE/KL 생성 CDR 전수 분석  (native trastuzumab 대비)")
print("=" * 72)
for name, cdrs in SETS.items():
    print(f"  {name}: {len(cdrs)} designs")

# ── 1) region별 native-identity (mean±std) ──
print("\n[1] region별 native-identity (%)  — 평균±표준편차")
print(f"  {'CDR':4s}{'len':>4s}{'CE id':>14s}{'KL id':>14s}")
region_id = {m: {} for m in SETS}
for n, L in segs:
    row = f"  {n:4s}{L:>4d}"
    for m, cdrs in SETS.items():
        ids = [sum(a == b for a, b in zip(split(c)[n], nat[n])) / L for c in cdrs]
        region_id[m][n] = ids
        row += f"{np.mean(ids)*100:>9.0f}±{np.std(ids)*100:<3.0f}"
    print(row)

# ── 2) L1 canonical: 길이 고정(11)→class 구조보존, 위치별 보존율 ──
print("\n[2] L1 (canonical Vκ1, len 11) — 위치별 native 보존율 (CE+KL 전체)")
allL1 = [split(c)["L1"] for cdrs in SETS.values() for c in cdrs]
print(f"  native L1: {nat['L1']}")
cons = "".join(nat["L1"][i] if np.mean([s[i] == nat["L1"][i] for s in allL1]) > 0.5 else "."
               for i in range(11))
print(f"  보존패턴 : {cons}   (대문자=과반보존, .=가변)")
for i, a in enumerate(nat["L1"]):
    frac = np.mean([s[i] == a for s in allL1])
    bar = "#" * int(frac * 20)
    print(f"    pos{i:>2d} {a}: {frac*100:>3.0f}% {bar}")

# ── 3) H1 Vernier(설계가 건드리는 framework 잔기) 분석 ──
print("\n[3] H1 N-말단 Vernier 잔기 (Kabat 28/29/30 = native N/I/K) — 설계가 뭘 넣나")
VPOS = [(0, "N", "28"), (1, "I", "29"), (2, "K", "30")]
for m, cdrs in SETS.items():
    print(f"  [{m}]")
    for idx, na, kab in VPOS:
        col = [split(c)["H1"][idx] for c in cdrs]
        from collections import Counter
        cnt = Counter(col).most_common(4)
        keep = sum(x == na for x in col) / len(col)
        top = " ".join(f"{aa}:{c}" for aa, c in cnt)
        print(f"    Kabat{kab} (native {na}): native보존 {keep*100:>3.0f}% | top: {top}")

# ── figure: region별 identity (CE vs KL) ──
fig, ax = plt.subplots(figsize=(9, 4.5))
x = np.arange(len(segs)); w = 0.38
for j, (m, col) in enumerate([("CE", "#1565c0"), ("KL", "#2e7d32")]):
    means = [np.mean(region_id[m][n]) * 100 for n, _ in segs]
    stds = [np.std(region_id[m][n]) * 100 for n, _ in segs]
    ax.bar(x + (j - 0.5) * w, means, w, yerr=stds, capsize=3, label=m, color=col, alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels([n for n, _ in segs])
ax.set_ylabel("identity to native (%)"); ax.set_ylim(0, 100)
ax.set_title("Generated CDR identity to native — by region (CE vs KL)\n"
             "L-CDRs/H1/H2 conserved (canonical), H3 free", fontsize=11)
ax.axhline(50, color="#aaa", ls="--", lw=0.7)
ax.legend(); plt.tight_layout()
plt.savefig("report/fig_ce_kl_cdr_identity.png", dpi=150, bbox_inches="tight")
print("\nsaved: report/fig_ce_kl_cdr_identity.png")
