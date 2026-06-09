"""Algorithm 11 baseline architecture + loss diagram (team meeting) — clean v2.
세로 backbone + 좌(seq prior)/우(structure) 2열 + 직각 화살표 + 손실 한 줄 정렬."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["axes.unicode_minus"] = False
fig, ax = plt.subplots(figsize=(16, 15))
ax.set_xlim(0, 16); ax.set_ylim(-0.6, 15.2); ax.axis("off")

C = dict(inp="#e3e3e3", model="#bbdefb", struct="#ffe0b2", seq="#c8e6c9",
         grad="#e1bee7", out="#fff59d", ours="#e53935")


def box(x, y, w, h, t, fc, fs=9.3, ec="#444", lw=1.2, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.1",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs,
            weight="bold" if bold else "normal")


def v(xc, y1, y2, c="#555", lw=1.7):            # 세로 화살표
    ax.add_patch(FancyArrowPatch((xc, y1), (xc, y2), arrowstyle="-|>", mutation_scale=15, color=c, lw=lw))


def elbow(pts, c="#555", lw=1.7):               # 직각 폴리라인 + 끝 화살표
    for i in range(len(pts) - 2):
        ax.plot([pts[i][0], pts[i + 1][0]], [pts[i][1], pts[i + 1][1]], color=c, lw=lw,
                solid_capstyle="round", zorder=1)
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=15, color=c, lw=lw))


# ── Title ──
ax.text(8, 14.8, "Algorithm 11 - Gradient-Guided Binder Sequence Optimization  (foundry baseline)",
        ha="center", fontsize=15.5, weight="bold")
ax.text(8, 14.35, "Redesign antibody scFv CDRs by ESMFold2 distogram gradients   (framework & antigen FIXED)",
        ha="center", fontsize=10.5, color="#555")

# ── Inputs ──
box(1.3, 13.2, 5.2, 0.75, "TARGET (antigen)  s_target\nonehot, FIXED", C["inp"], 9.3)
box(9.5, 13.2, 5.2, 0.75, "BINDER prompt = scFv (VH-linker-VL)\nframework FIXED + CDR mutable(#), no Cys", C["inp"], 9.3)
box(3.6, 12.05, 8.8, 0.7,
    "Init logits  x in R[L x 20]   (fixed=10, mutable~N(0,1e-4), Cys=-1e6)   +   grad mask m", C["inp"], 9.3)
elbow([(3.9, 13.2), (3.9, 12.78), (8, 12.78), (8, 12.75)]);
elbow([(12.1, 13.2), (12.1, 12.78), (8, 12.78)])

# ── Loop container ──
ax.add_patch(FancyBboxPatch((0.4, 0.95), 15.2, 10.7, boxstyle="round,pad=0.1,rounding_size=0.2",
                            fc="#fbfbfb", ec="#1565c0", lw=2.2, ls=(0, (6, 4))))
ax.text(0.75, 11.05, "for  k = 1 ... K=150     |     cosine anneal  T_k: 1 -> 0.01,   alpha_k = 0.1 * T_k",
        fontsize=11.5, weight="bold", color="#1565c0")
v(8, 12.05, 11.12)

# ── Backbone top (center) ──
box(6.7, 10.45, 2.6, 0.6, "logits  x  [L,20]", C["grad"], 9.3, bold=True)
v(8, 10.45, 10.0)
box(5.6, 9.3, 4.8, 0.65, "softmax(x / T_k)  ->  soft binder [L,20]", C["grad"], 9.3, bold=True)

# branch bus from soft binder
ax.plot([8, 8], [9.3, 8.95], color="#555", lw=1.7, zorder=1)              # down to bus
SEQX, STRX = 3.6, 10.8
ax.plot([SEQX, STRX], [8.95, 8.95], color="#555", lw=1.7, zorder=1)       # horizontal bus
ax.text(8, 9.12, "soft binder (sequence)", ha="center", fontsize=8, color="#777", style="italic")

# ── Column headers ──
ax.text(3.55, 8.6, "SEQUENCE PRIORS (naturalness)", ha="center", fontsize=9.5, weight="bold", color="#2e7d32")
ax.text(11.4, 8.6, "STRUCTURE PATH (ESMFold2)", ha="center", fontsize=9.5, weight="bold", color="#1565c0")

# ── Structure column (right) ──
elbow([(STRX, 8.95), (STRX, 8.25)])
box(8.5, 7.5, 4.6, 0.72, "concat [onehot(target); soft]\n->  soft complex", C["model"], 9.0)
v(STRX, 7.5, 7.05)
box(8.5, 6.25, 4.6, 0.78, "ESMFold2  F  (ESMC-6B trunk)\nT>=0.05: distogram only / T<0.05: +conf head", C["model"], 8.7, bold=True)
box(13.5, 6.28, 2.0, 0.72, "* low-T\nreal ipTM_k -> b*", C["seq"], 8.2, ec=C["ours"], lw=1.8)
ax.add_patch(FancyArrowPatch((13.1, 6.64), (13.5, 6.64), arrowstyle="-|>", mutation_scale=13, color=C["ours"], lw=1.6))
v(STRX, 6.25, 5.85)
box(9.7, 5.2, 2.2, 0.6, "distogram  D_k", C["model"], 9.0, bold=True)

# distogram -> 3 struct losses (bus)
ax.plot([STRX, STRX], [5.2, 4.85], color="#a05a00", lw=1.6, zorder=1)
ax.plot([8.45, 13.55], [4.85, 4.85], color="#a05a00", lw=1.6, zorder=1)
for cx in (8.45, 11.0, 13.55):
    v(cx, 4.85, 4.18, c="#a05a00")

# ── Sequence column (left): soft binder -> two priors ──
elbow([(SEQX, 8.95), (SEQX, 4.85)], c="#2e7d32")
ax.plot([2.05, 5.15], [4.85, 4.85], color="#2e7d32", lw=1.6, zorder=1)
for cx in (2.05, 5.15):
    v(cx, 4.85, 4.18, c="#2e7d32")

# ── Loss band (one row) ──
box(0.8, 3.25, 2.5, 0.9, "L_LM\nESMC masked-PPL\nM=4, lam_LM=0.05", C["seq"], 8.2)
box(3.9, 3.25, 2.5, 0.9, "* L_comp\ncomposition PSSM (CE)\nlam_comp=0.5  (ours)", C["seq"], 8.2, ec=C["ours"], lw=1.8)
box(7.3, 3.25, 2.3, 0.9, "L_inter (lam=0.5)\nCDR <-> epitope\ncontacts", C["struct"], 8.2)
box(9.85, 3.25, 2.3, 0.9, "L_intra (lam=0.5)\nscFv internal\ncontacts", C["struct"], 8.2)
box(12.4, 3.25, 2.3, 0.9, "L_glob (lam=0.2)\nglobularity\n(compact)", C["struct"], 8.2)

# losses -> gradient box
GY = 2.05
for cx in (2.05, 5.15, 8.45, 11.0, 13.55):
    v(cx, 3.25, GY + 0.62)
box(0.8, GY, 14.4, 0.62,
    "grad each loss -> per-component normalize  g_hat = sqrt(n_mut)*(g . m)/||g . m||    ->   "
    "g = g_hat_struct + lam_LM * g_hat_LM + lam_comp * g_hat_comp", C["grad"], 8.8)
v(8, GY, 1.55)
box(5.3, 0.98, 5.4, 0.5, "SGD :   x  <-  x  -  alpha_k * g", C["grad"], 9.5, bold=True)

# loop-back (far left margin)
elbow([(5.3, 1.23), (1.0, 1.23), (1.0, 10.75), (6.7, 10.75)], c="#1565c0", lw=1.8)
ax.text(0.78, 6.0, "update x  (loop)", rotation=90, va="center", fontsize=8.5, color="#1565c0")

# ── Output + Post (below loop) ──
v(8, 0.95, 0.62)
box(3.4, -0.15, 4.6, 0.62, "OUTPUT  b*  (best-ipTM design)\nCDR -> graft onto scFv", C["out"], 9.0, bold=True)
ax.add_patch(FancyArrowPatch((8.0, 0.16), (8.7, 0.16), arrowstyle="-|>", mutation_scale=14, color="#555", lw=1.7))
box(8.7, -0.15, 6.5, 0.62, "POST: 4-critic ipSAE ensemble ranking\n(4 independent Experimental critics) -> final", C["model"], 9.0)

# ── Legend (top-right) ──
lx, ly = 0.6, -0.45
for lab, col in [("input/init", C["inp"]), ("model (ESMFold2/ESMC)", C["model"]),
                 ("structure loss", C["struct"]), ("sequence prior", C["seq"]),
                 ("gradient/update", C["grad"]), ("* OUR add / faithful", "#ffffff")]:
    ec = C["ours"] if lab.startswith("*") else "#444"
    ax.add_patch(FancyBboxPatch((lx, ly), 0.32, 0.22, boxstyle="round,pad=0.02", fc=col, ec=ec, lw=1.6))
    ax.text(lx + 0.42, ly + 0.11, lab, fontsize=8.3, va="center"); lx += 2.55

plt.savefig("report/fig0_architecture.png", dpi=140, bbox_inches="tight")
print("saved: report/fig0_architecture.png")
