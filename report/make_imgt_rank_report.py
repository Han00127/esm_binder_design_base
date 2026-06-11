"""IMGT 후보 ipSAE 랭킹 시각화 + native-identity. (2026-06-12)"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir("/home/kyeongtak/structure_projects/esm_binder_design_base")
plt.rcParams["axes.unicode_minus"] = False

# native IMGT-CDR (region별)
NAT = {"H1": "GFNIKDTY", "H2": "IYPTNGYT", "H3": "SRWGGDGFYAMDY",
       "L1": "QDVNTA", "L2": "SAS", "L3": "QQHYTTPPT"}
SEGS = [("H1", 8), ("H2", 8), ("H3", 13), ("L1", 6), ("L2", 3), ("L3", 9)]
CKS = ["exp_full_2021", "exp_full_2025", "exp_fast_2021", "exp_fast_2025"]
NATCAT = "".join(NAT[c] for c, _ in SEGS)

r = json.load(open("runs/rank_imgt/ranked.json"))
r = sorted(r, key=lambda x: -x["avg_ipsae"])


def split(cdr):
    out, off = {}, 0
    for c, L in SEGS:
        out[c] = cdr[off:off + L]; off += L
    return out


# ── 콘솔: identity ──
print("=" * 74)
print(" IMGT 후보 — native-identity (region별) + avg_ipsae")
print("=" * 74)
print(f"  {'cand':5s}{'ipSAE':>7s}{'전체id':>7s} | " + " ".join(f"{c:>5s}" for c, _ in SEGS))
ident_rows = {}
for x in r:
    p = split(x["cdr"])
    tot = sum(a == b for a, b in zip(x["cdr"], NATCAT)) / len(NATCAT)
    reg = {c: sum(a == b for a, b in zip(p[c], NAT[c])) / L for c, L in SEGS}
    ident_rows[x["name"]] = (tot, reg)
    print(f"  {x['name']:5s}{x['avg_ipsae']:>7.3f}{tot*100:>6.0f}% | " +
          " ".join(f"{reg[c]*100:>4.0f}%" for c, _ in SEGS))

# ── Figure: (좌) ipSAE 랭킹 + per-critic, (우) region별 identity heatmap ──
fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5.2))
names = [x["name"] for x in r]
avg = [x["avg_ipsae"] for x in r]
xpos = np.arange(len(r))
bars = axA.bar(xpos, avg, color="#1565c0", alpha=0.85, width=0.6, zorder=2)
for i, x in enumerate(r):                      # per-critic dots
    for ck in CKS:
        v = x["per_critic"].get(ck)
        if v is not None:
            axA.scatter(i, v, c="#e53935", s=22, zorder=4, alpha=0.8)
axA.scatter([], [], c="#e53935", s=22, label="per-critic (4종)")
for i, v in enumerate(avg):
    axA.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9, weight="bold")
axA.axhline(0.167, color="#999", ls="--", lw=0.8, label="Kabat A/B best (0.167)")
axA.set_xticks(xpos); axA.set_xticklabels(names)
axA.set_ylabel("avg ipSAE (4-critic)"); axA.set_ylim(0, 1.0)
axA.set_title("IMGT candidates — ipSAE ranking\n(bar=4-critic mean, red dots=per-critic)", fontsize=11)
axA.legend(fontsize=8, loc="upper right")

mat = np.array([[ident_rows[n][1][c] for c, _ in SEGS] for n in names])
im = axB.imshow(mat, cmap="YlGn", vmin=0, vmax=1, aspect="auto")
axB.set_xticks(range(len(SEGS))); axB.set_xticklabels([c for c, _ in SEGS])
axB.set_yticks(range(len(names))); axB.set_yticklabels(names)
for i in range(len(names)):
    for j in range(len(SEGS)):
        axB.text(j, i, f"{mat[i,j]*100:.0f}", ha="center", va="center", fontsize=9,
                 color="white" if mat[i, j] > 0.5 else "#333")
axB.set_title("identity to native by region (%)", fontsize=11)
fig.colorbar(im, ax=axB, shrink=0.7, label="identity")
plt.tight_layout()
plt.savefig("report/fig_imgt_rank.png", dpi=150, bbox_inches="tight")
print("\nsaved: report/fig_imgt_rank.png")
