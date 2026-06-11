"""gradient 성분별 정규화 개념도: raw(큰 항 독점) vs 정규화(λ로 균형). 영문 라벨(tofu 방지)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir("/home/kyeongtak/structure_projects/esm_binder_design_base")

# 같은 '방향'의 두 gradient (크기만 다름)
gs_dir = np.array([4.9, 0.9]); gs_dir = gs_dir / np.linalg.norm(gs_dir)
gc_dir = np.array([-0.5, 1.0]); gc_dir = gc_dir / np.linalg.norm(gc_dir)

# raw 크기: 구조는 크고 composition은 작음
gs_raw = gs_dir * 4.6
gc_raw = gc_dir * 0.45
sum_raw = gs_raw + gc_raw                      # ≈ 구조 방향 (comp 묻힘)

# 정규화: 둘 다 같은 norm(=L, √n 대리) → λ 가중
L = 3.0
lam = 0.75
gs_n = gs_dir * L
gc_n = gc_dir * L
sum_n = gs_n + lam * gc_n                       # comp가 실제로 방향을 꺾음

BLUE, GREEN, BLACK = "#1565c0", "#2e7d32", "#222"


def draw(ax, gs, gc, ssum, title, comp_label, note):
    ax.axhline(0, color="#ddd", lw=0.8, zorder=0); ax.axvline(0, color="#ddd", lw=0.8, zorder=0)
    kw = dict(angles="xy", scale_units="xy", scale=1, width=0.013)
    ax.quiver(0, 0, *gs, color=BLUE, zorder=3, **kw)
    ax.quiver(0, 0, *gc, color=GREEN, zorder=3, **kw)
    ax.quiver(0, 0, *ssum, color=BLACK, zorder=4, **kw)
    ax.annotate("g_struct", gs * 1.02, color=BLUE, fontsize=10, weight="bold")
    ax.annotate(comp_label, gc + np.array([0.05, 0.12]), color=GREEN, fontsize=10, weight="bold")
    ax.annotate("combined\n(update)", ssum * 1.03 + np.array([0.1, -0.1]), color=BLACK,
                fontsize=10, weight="bold")
    ax.set_title(title, fontsize=12, weight="bold")
    ax.text(0.5, -0.13, note, transform=ax.transAxes, ha="center", va="top",
            fontsize=9.5, color="#555")
    ax.set_xlim(-2.2, 5.4); ax.set_ylim(-1.2, 4.2); ax.set_aspect("equal"); ax.axis("off")


fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.6))
draw(axA, gs_raw, gc_raw, sum_raw, "Without normalization  (raw gradients)", "g_comp",
     "‖g_struct‖ >> ‖g_comp‖  →  combined ≈ g_struct\ncomposition is drowned out (λ can't fix it)")
draw(axB, gs_n, lam * gc_n, sum_n, "Per-component normalization  +  λ", "λ·ĝ_comp",
     "both rescaled to ‖·‖ = √n, then weighted by λ\n→ λ truly controls the balance; comp steers the update")
fig.suptitle("Why per-component gradient normalization?   ĝ = √n · (g⊙m) / ‖g⊙m‖   (Alg11 line24-25)",
             fontsize=13, weight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("report/fig_gradnorm.png", dpi=150, bbox_inches="tight")
print("saved: report/fig_gradnorm.png")
