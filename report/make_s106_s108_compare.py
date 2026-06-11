"""s106 / s108 생성 CDR vs native trastuzumab — region별(H1/H2/H3/L1/L2/L3) 비교.
fig4b 스타일: 방향족(W/Y/F)=빨강. + native 대비 보존(회색)/변경(검정) 구분, region별 identity·arom%.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

os.chdir("/home/kyeongtak/structure_projects/esm_binder_design_base")
plt.rcParams["axes.unicode_minus"] = False

# region 경계 (q_target.npz cdr_names 기준): H1 8 / H2 12 / H3 11 / L1 11 / L2 7 / L3 9 = 58
segs = [("H1", 8), ("H2", 12), ("H3", 11), ("L1", 11), ("L2", 7), ("L3", 9)]
bounds, off = {}, 0
for name, L in segs:
    bounds[name] = (off, off + L); off += L
TOTAL = off

NATIVE = "NIKDTYIHIYPTNGYTRYADWGGDGFYAMDYRASQDVNTAVASASFLYSQQHYTTPPT"
DESIGNS = [
    ("native trastuzumab", NATIVE),
    ("s106", "TFSSYYMSIYPSGGSTNYADDGRGGYYGFDVRASQSLSSNLYAASSRASQQSNSPPYT"),
    ("s108", "TFSSYYMSIYSSGGSTYYADGGRYGGYGFDYRASQSLSSSLYGASSRASQQYNSPPLT"),
]
for nm, s in DESIGNS:
    assert len(s) == TOTAL, f"{nm} len {len(s)} != {TOTAL}"


def split_cdr(c):
    return {name: c[a:b] for name, (a, b) in bounds.items()}


# ── 콘솔 비교표 ──
nat = split_cdr(NATIVE)
print("=" * 70)
print(" s106 / s108  vs  native trastuzumab — region별 비교")
print("=" * 70)
for label, cdr in DESIGNS:
    p = split_cdr(cdr)
    print(f"\n[{label}]")
    for name, L in segs:
        s = p[name]
        arom = sum(c in "WYF" for c in s) / L
        if label == "native trastuzumab":
            print(f"  {name:3s} {s:<13s}  arom {arom:>3.0%}")
        else:
            ident = sum(a == b for a, b in zip(s, nat[name])) / L
            print(f"  {name:3s} {s:<13s}  arom {arom:>3.0%} | native대비 동일 {ident:>3.0%}  (native {nat[name]})")

# ── figure (fig4b 스타일) ──
GAP = 2.0
xpos, x = {}, 1.0
for name, L in segs:
    xpos[name] = x; x += L + GAP
XMAX = x
rows = DESIGNS
fig, ax = plt.subplots(figsize=(max(13, XMAX * 0.23), 1.4 + 0.95 * len(rows)))
ax.set_xlim(0, XMAX); ax.set_ylim(0, len(rows) + 2.4); ax.axis("off")
ax.text(XMAX / 2, len(rows) + 1.9, "s106 / s108  vs  native trastuzumab — CDR-by-CDR",
        ha="center", fontsize=14, weight="bold")
ax.text(XMAX / 2, len(rows) + 1.35,
        "aromatic W/Y/F = red    ·    same as native = grey,  changed = black  (design rows)",
        ha="center", fontsize=9, color="#666")

yhead = len(rows) + 0.65
for name, L in segs:
    x0 = xpos[name]
    ax.add_patch(Rectangle((x0 - 0.5, 0.3), L, len(rows) + 0.4, fc="#f2f2f2", ec="none", zorder=0))
    ax.text(x0 + L / 2 - 0.5, yhead, f"{name}\n(len {L})", ha="center", va="center",
            fontsize=11, weight="bold", color="#333")

COL = {"native trastuzumab": "#7e57c2", "s106": "#1565c0", "s108": "#00838f"}
for r, (label, cdr) in enumerate(rows):
    y = len(rows) - r
    ax.text(-0.3, y, label, ha="right", va="center", fontsize=10, weight="bold", color=COL[label])
    parts = split_cdr(cdr)
    is_native = label == "native trastuzumab"
    for name, L in segs:
        x0 = xpos[name]; seq = parts[name]; nseq = nat[name]
        for i, aa in enumerate(seq):
            arom = aa in "WYF"
            if arom:
                col, w = "#e53935", "bold"
            elif (not is_native) and aa == nseq[i]:
                col, w = "#bbbbbb", "normal"          # native와 동일 = 흐리게
            else:
                col, w = "#222", "normal"
            ax.text(x0 + i, y, aa, ha="center", va="center", family="monospace",
                    fontsize=11, weight=w, color=col)
        af = sum(c in "WYF" for c in seq) / L
        if is_native:
            ax.text(x0 + L / 2 - 0.5, y - 0.42, f"arom {af:.0%}", ha="center", fontsize=7,
                    color="#e53935" if af > 0.25 else "#999")
        else:
            idn = sum(a == b for a, b in zip(seq, nseq)) / L
            ax.text(x0 + L / 2 - 0.5, y - 0.42, f"arom {af:.0%} · id {idn:.0%}", ha="center",
                    fontsize=7, color="#666")

plt.tight_layout()
out = "report/fig_s106_s108_cdr.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved:", out)
