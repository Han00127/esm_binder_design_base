"""CDR별(H1/H2/H3/L1/L2/L3) 서열 비교 figure. 방향족(W/Y/F)=빨강 강조."""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams["axes.unicode_minus"] = False
ESM = "/home/kyeongtak/structure_projects/esm_binder_design_base"
os.chdir(ESM)
import yaml

cfg = yaml.safe_load(open("configs/trastuzumab_her2.yaml"))
# CDR 세그먼트(이름,길이) — concat 순서 = heavy(H1,H2,H3) → light(L1,L2,L3), 각 range(s,e)
segs = []
for n, (s, e) in cfg["antibody"]["heavy"]["cdr_ranges"].items():
    segs.append((f"H{n[-1]}", e - s))
for n, (s, e) in cfg["antibody"]["light"]["cdr_ranges"].items():
    segs.append((f"L{n[-1]}", e - s))
# 경계 인덱스
bounds = {}
off = 0
for name, L in segs:
    bounds[name] = (off, off + L); off += L
TOTAL = off
print("CDR segments:", segs, "total", TOTAL)


def split_cdr(cdr):
    return {name: cdr[a:b] for name, (a, b) in bounds.items()}


def load1(pat):
    for f in sorted(glob.glob(pat)):
        c = json.load(open(f)).get("candidates", [])
        if c:
            return c[0]["cdr"]
    return None


NATIVE = "NIKDTYIHIYPTNGYTRYADWGGDGFYAMDYRASQDVNTAVASASFLYSQQHYTTPPT"
rows = []
nc = load1("runs/baseline_proxy/cand_g*.json")
if nc:
    rows.append(("no-comp (baseline)", nc, "#616161"))
kl = load1("runs/batch30_kl.json")
if kl:
    rows.append(("KL-comp", kl, "#2e7d32"))
ce = load1("runs/batch_ce_g*.json")
if ce:
    rows.append(("CE-comp", ce, "#1565c0"))
rows.append(("native trastuzumab", NATIVE, "#7e57c2"))

# ── 레이아웃: CDR 컬럼(길이 비례) + 디자인 행 ──
GAP = 2.0
xpos, x = {}, 1.0
for name, L in segs:
    xpos[name] = x; x += L + GAP
XMAX = x

fig, ax = plt.subplots(figsize=(max(13, XMAX * 0.23), 1.2 + 0.9 * len(rows)))
ax.set_xlim(0, XMAX); ax.set_ylim(0, len(rows) + 2.2); ax.axis("off")
ax.text(XMAX / 2, len(rows) + 1.7, "CDR-by-CDR comparison  (aromatic W/Y/F = red)",
        ha="center", fontsize=14, weight="bold")

yhead = len(rows) + 0.9
# CDR 헤더 + 세로 구분 음영
for name, L in segs:
    x0 = xpos[name]
    ax.add_patch(Rectangle((x0 - 0.5, 0.3), L, len(rows) + 0.5, fc="#f2f2f2", ec="none", zorder=0))
    ax.text(x0 + L / 2 - 0.5, yhead, f"{name}\n(len {L})", ha="center", va="center",
            fontsize=11, weight="bold", color="#333")

for r, (label, cdr, col) in enumerate(rows):
    y = len(rows) - r            # 위에서부터
    ax.text(-0.3, y, label, ha="right", va="center", fontsize=9.5, weight="bold", color=col)
    parts = split_cdr(cdr)
    for name, L in segs:
        x0 = xpos[name]
        seq = parts[name]
        for i, aa in enumerate(seq):
            arom = aa in "WYF"
            ax.text(x0 + i, y, aa, ha="center", va="center", family="monospace",
                    fontsize=11, weight="bold" if arom else "normal",
                    color="#e53935" if arom else "#222")
        # 영역별 방향족 분율
        af = sum(c in "WYF" for c in seq) / max(1, len(seq))
        ax.text(x0 + L / 2 - 0.5, y - 0.42, f"{af:.0%}", ha="center", va="center",
                fontsize=7.5, color="#e53935" if af > 0.25 else "#999")

ax.text(-0.3, 0.15, "(% = aromatic fraction per CDR)", ha="left", fontsize=8, color="#999", style="italic")
plt.tight_layout()
plt.savefig("report/fig4b_cdr_by_region.png", dpi=150, bbox_inches="tight")
print("saved: report/fig4b_cdr_by_region.png")
# 콘솔 요약
for label, cdr, _ in rows:
    p = split_cdr(cdr)
    print(f"\n{label}:")
    for name, L in segs:
        s = p[name]; af = sum(c in 'WYF' for c in s) / len(s)
        print(f"  {name}: {s}  (arom {af:.0%})")
