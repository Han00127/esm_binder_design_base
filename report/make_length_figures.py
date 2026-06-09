"""길이층화 PSSM 시각화: (1) CDR별 길이 분포, (2) 길이별 PSSM 비교(H3 예시)."""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
for _f in glob.glob("/usr/share/fonts/google-noto-cjk/NotoSansCJK-*.ttc"):
    try:
        from matplotlib import font_manager as _fm
        _fm.fontManager.addfont(_f)
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False

ESM = "/home/kyeongtak/structure_projects/esm_binder_design_base"
os.chdir(ESM)
meta = json.load(open("data/length_pssm_stats.json"))
npz = np.load("data/length_pssm.npz")
AA = meta["aa_order"]
hist = meta["length_hist"]
CDRS = ["H1", "H2", "H3", "L1", "L2", "L3"]

# ── Fig A: CDR별 길이 분포 ──
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for ax, cdr in zip(axes.flat, CDRS):
    if cdr not in hist:
        ax.set_visible(False); continue
    h = {int(k): v for k, v in hist[cdr].items()}
    tot = sum(h.values())
    Ls = sorted(h)
    ax.bar(Ls, [100 * h[L] / tot for L in Ls], color="#5c9", edgecolor="k", lw=0.3)
    mode = max(h, key=h.get)
    ax.set_title(f"{cdr}  (mode={mode}, range {min(Ls)}-{max(Ls)})")
    ax.set_xlabel("CDR length"); ax.set_ylabel("% of antibodies")
    ax.grid(axis="y", alpha=0.3)
fig.suptitle(f"CDR length distribution (natural Abs, n={meta['n_seq']:,}; OAS+TheraSAbDab)",
             fontsize=13, weight="bold")
plt.tight_layout(); plt.savefig("report/figL1_length_dist.png", dpi=130); plt.close()

# ── Fig B: H3 길이별 PSSM 비교 (같은 CDR이라도 길이마다 분포 다름) ──
h3_lengths = sorted(int(k.split("__")[1]) for k in npz.files if k.startswith("H3__"))
# 대표 3개 길이 (짧음/중간/김)
if h3_lengths:
    pick = [h3_lengths[0], h3_lengths[len(h3_lengths) // 2], h3_lengths[-1]]
    fig, axes = plt.subplots(len(pick), 1, figsize=(12, 2.4 * len(pick)))
    if len(pick) == 1:
        axes = [axes]
    for ax, L in zip(axes, pick):
        pssm = npz[f"H3__{L}"]                      # [L,20]
        im = ax.imshow(pssm.T, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_yticks(range(20)); ax.set_yticklabels(list(AA), fontsize=7)
        ax.set_title(f"H3 length={L}  (per-position natural distribution)")
        ax.set_xlabel("position (0..L-1)")
    fig.colorbar(im, ax=axes, label="P(AA|pos)", fraction=0.02)
    fig.suptitle("H3 length-stratified PSSM: distribution differs by length (apex shifts)",
                 fontsize=12, weight="bold")
    plt.savefig("report/figL2_h3_pssm_by_length.png", dpi=130, bbox_inches="tight"); plt.close()

# ── Fig L3: 방향족(W/Y/F) 함량 — 길이별 ──
def arom_frac_by_len(cdr):
    ah = meta.get("arom_hist", {}).get(cdr, {})
    lh = meta["length_hist"].get(cdr, {})
    Ls, mean, std = [], [], []
    for L in sorted(int(x) for x in ah):
        hh = ah[str(L)]
        ks = np.array([int(k) for k in hh], float); cs = np.array([hh[k] for k in hh], float)
        n = cs.sum()
        if n < 30:
            continue
        fr = ks / L
        m = (fr * cs).sum() / n
        v = ((fr - m) ** 2 * cs).sum() / n
        Ls.append(L); mean.append(m); std.append(np.sqrt(v))
    return np.array(Ls), np.array(mean), np.array(std)


def wyf_frac_by_len(cdr):
    ws = meta.get("wyf_sum", {}).get(cdr, {}); lh = meta["length_hist"].get(cdr, {})
    Ls, W, Y, F = [], [], [], []
    for L in sorted(int(x) for x in ws):
        n = lh[str(L)]
        if n < 30:
            continue
        w, y, f = ws[str(L)]
        Ls.append(L); W.append(w / (n * L)); Y.append(y / (n * L)); F.append(f / (n * L))
    return np.array(Ls), np.array(W), np.array(Y), np.array(F)


fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for cdr, c in zip(CDRS, plt.cm.tab10.colors):
    Ls, m, s = arom_frac_by_len(cdr)
    if len(Ls):
        axes[0].plot(Ls, m, marker="o", label=cdr, color=c)
        axes[0].fill_between(Ls, m - s, m + s, color=c, alpha=0.12)
axes[0].axhline(0.195, ls="--", c="gray", label="overall ~0.195")
axes[0].set_xlabel("CDR length"); axes[0].set_ylabel("aromatic (W+Y+F) per-residue fraction")
axes[0].set_title("Aromatic content vs length, by CDR (band=std)")
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
Ls, W, Y, F = wyf_frac_by_len("H3")
if len(Ls):
    axes[1].plot(Ls, Y, marker="o", label="Tyr (Y)", color="#d62728")
    axes[1].plot(Ls, W, marker="s", label="Trp (W)", color="#1f77b4")
    axes[1].plot(Ls, F, marker="^", label="Phe (F)", color="#2ca02c")
axes[1].set_xlabel("H3 length"); axes[1].set_ylabel("per-residue fraction")
axes[1].set_title("H3: Tyr / Trp / Phe vs length"); axes[1].legend(); axes[1].grid(alpha=0.3)
fig.suptitle("Natural antibody CDR aromatic content by length (OAS+TheraSAbDab)",
             fontsize=13, weight="bold")
plt.tight_layout(); plt.savefig("report/figL3_aromatic_by_length.png", dpi=130); plt.close()

# 요약 출력
print("\n=== 방향족 함량 (per-residue) by length ===")
for cdr in CDRS:
    Ls, m, s = arom_frac_by_len(cdr)
    if len(Ls):
        print(f"  {cdr}: len {Ls.min()}-{Ls.max()}, 방향족 {m.min():.2f}~{m.max():.2f} "
              f"(전체평균 {(m).mean():.2f})")
print("저장: report/figL1_length_dist.png, figL2_h3_pssm_by_length.png, figL3_aromatic_by_length.png")
print("H3 보유 길이:", h3_lengths)
for cdr in CDRS:
    if cdr in hist:
        h = {int(k): v for k, v in hist[cdr].items()}
        tot = sum(h.values()); mode = max(h, key=h.get)
        print(f"  {cdr}: mode={mode} ({100*h[mode]/tot:.0f}%), range {min(h)}-{max(h)}")
