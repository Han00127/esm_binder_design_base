"""IMGT 후보 s0~s4 vs native — region별 서열 비교 (fig4b 스타일). 방향족=빨강, native동일=회색, 변경=검정."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

os.chdir("/home/kyeongtak/structure_projects/esm_binder_design_base")
plt.rcParams["axes.unicode_minus"] = False

NAT = {"H1": "GFNIKDTY", "H2": "IYPTNGYT", "H3": "SRWGGDGFYAMDY",
       "L1": "QDVNTA", "L2": "SAS", "L3": "QQHYTTPPT"}
SEGS = [("H1", 8), ("H2", 8), ("H3", 13), ("L1", 6), ("L2", 3), ("L3", 9)]
bounds, off = {}, 0
for n, L in SEGS:
    bounds[n] = (off, off + L); off += L
NATCAT = "".join(NAT[n] for n, _ in SEGS)

r = sorted(json.load(open("runs/rank_imgt/ranked.json")), key=lambda x: -x["avg_ipsae"])
rows = [("native trastuzumab", NATCAT, None)]
for x in r:
    rows.append((f"{x['name']} (ipSAE {x['avg_ipsae']:.2f})", x["cdr"], x["avg_ipsae"]))


def split(c):
    return {n: c[a:b] for n, (a, b) in bounds.items()}


GAP = 2.0
xpos, x = {}, 1.0
for n, L in SEGS:
    xpos[n] = x; x += L + GAP
XMAX = x
fig, ax = plt.subplots(figsize=(max(13, XMAX * 0.24), 1.5 + 0.85 * len(rows)))
ax.set_xlim(0, XMAX); ax.set_ylim(0, len(rows) + 2.3); ax.axis("off")
ax.text(XMAX / 2, len(rows) + 1.8, "IMGT candidates vs native — CDR-by-CDR  (ranked by ipSAE)",
        ha="center", fontsize=14, weight="bold")
ax.text(XMAX / 2, len(rows) + 1.25,
        "aromatic W/Y/F = red   ·   same as native = grey,  changed = black",
        ha="center", fontsize=9, color="#666")

yhead = len(rows) + 0.6
for n, L in SEGS:
    x0 = xpos[n]
    ax.add_patch(Rectangle((x0 - 0.5, 0.3), L, len(rows) + 0.35, fc="#f2f2f2", ec="none", zorder=0))
    ax.text(x0 + L / 2 - 0.5, yhead, f"{n}\n(len {L})", ha="center", va="center",
            fontsize=11, weight="bold", color="#333")

for ri, (label, cdr, ips) in enumerate(rows):
    y = len(rows) - ri
    col = "#7e57c2" if ips is None else ("#1565c0" if ips >= 0.6 else "#888")
    ax.text(-0.3, y, label, ha="right", va="center", fontsize=9.5, weight="bold", color=col)
    parts = split(cdr); is_nat = ips is None
    for n, L in SEGS:
        x0 = xpos[n]; seq = parts[n]; nseq = NAT[n]
        for i, aa in enumerate(seq):
            if aa in "WYF":
                c, w = "#e53935", "bold"
            elif (not is_nat) and aa == nseq[i]:
                c, w = "#c0c0c0", "normal"
            else:
                c, w = "#222", "normal"
            ax.text(x0 + i, y, aa, ha="center", va="center", family="monospace",
                    fontsize=11, weight=w, color=c)
        if not is_nat:
            idn = sum(a == b for a, b in zip(seq, nseq)) / L
            ax.text(x0 + L / 2 - 0.5, y - 0.4, f"{idn:.0%}", ha="center", fontsize=6.5, color="#888")

plt.tight_layout()
plt.savefig("report/fig_imgt_seq_compare.png", dpi=150, bbox_inches="tight")
print("saved: report/fig_imgt_seq_compare.png")
