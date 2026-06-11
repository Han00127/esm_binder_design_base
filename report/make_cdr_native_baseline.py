"""CDR별 비교 — native trastuzumab vs baseline(no-comp) 두 행만. (fig4b 스타일: 방향족 W/Y/F=빨강)"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import yaml

ESM = "/home/kyeongtak/structure_projects/esm_binder_design_base"
os.chdir(ESM)
plt.rcParams["axes.unicode_minus"] = False

cfg = yaml.safe_load(open("configs/trastuzumab_her2.yaml"))
segs = []
for n, (s, e) in cfg["antibody"]["heavy"]["cdr_ranges"].items():
    segs.append((f"H{n[-1]}", e - s))
for n, (s, e) in cfg["antibody"]["light"]["cdr_ranges"].items():
    segs.append((f"L{n[-1]}", e - s))
bounds, off = {}, 0
for name, L in segs:
    bounds[name] = (off, off + L); off += L


def split_cdr(cdr):
    return {name: cdr[a:b] for name, (a, b) in bounds.items()}


def load1(pat):
    for f in sorted(glob.glob(pat)):
        c = json.load(open(f)).get("candidates", [])
        if c:
            return c[0]["cdr"]
    return None


NATIVE = "NIKDTYIHIYPTNGYTRYADWGGDGFYAMDYRASQDVNTAVASASFLYSQQHYTTPPT"
baseline = load1("runs/baseline_proxy/cand_g*.json")
assert baseline, "baseline 후보 없음"

# 행: baseline(위) → native(아래, 참조)
rows = [("baseline (no-comp)", baseline, "#616161"),
        ("native trastuzumab", NATIVE, "#7e57c2")]

GAP = 2.0
xpos, x = {}, 1.0
for name, L in segs:
    xpos[name] = x; x += L + GAP
XMAX = x

fig, ax = plt.subplots(figsize=(max(13, XMAX * 0.23), 1.2 + 1.0 * len(rows)))
ax.set_xlim(0, XMAX); ax.set_ylim(0, len(rows) + 2.2); ax.axis("off")
ax.text(XMAX / 2, len(rows) + 1.7, "CDR-by-CDR:  native trastuzumab  vs  baseline (no-comp)",
        ha="center", fontsize=14, weight="bold")
ax.text(XMAX / 2, len(rows) + 1.15, "aromatic W/Y/F = red    ·    (% = aromatic fraction per CDR)",
        ha="center", fontsize=9, color="#666")

yhead = len(rows) + 0.7
for name, L in segs:
    x0 = xpos[name]
    ax.add_patch(Rectangle((x0 - 0.5, 0.3), L, len(rows) + 0.45, fc="#f2f2f2", ec="none", zorder=0))
    ax.text(x0 + L / 2 - 0.5, yhead, f"{name}\n(len {L})", ha="center", va="center",
            fontsize=11, weight="bold", color="#333")

for r, (label, cdr, col) in enumerate(rows):
    y = len(rows) - r
    ax.text(-0.3, y, label, ha="right", va="center", fontsize=10.5, weight="bold", color=col)
    parts = split_cdr(cdr)
    for name, L in segs:
        x0 = xpos[name]; seq = parts[name]
        for i, aa in enumerate(seq):
            arom = aa in "WYF"
            ax.text(x0 + i, y, aa, ha="center", va="center", family="monospace",
                    fontsize=12, weight="bold" if arom else "normal",
                    color="#e53935" if arom else "#222")
        af = sum(c in "WYF" for c in seq) / max(1, len(seq))
        ax.text(x0 + L / 2 - 0.5, y - 0.42, f"{af:.0%}", ha="center", va="center",
                fontsize=8, color="#e53935" if af > 0.25 else "#999")

plt.tight_layout()
plt.savefig("report/fig4b_native_baseline.png", dpi=150, bbox_inches="tight")
print("saved: report/fig4b_native_baseline.png")
for label, cdr, _ in rows:
    p = split_cdr(cdr)
    print(f"\n{label}:")
    for name, L in segs:
        s = p[name]; af = sum(c in "WYF" for c in s) / len(s)
        print(f"  {name}: {s}  (arom {af:.0%})")
