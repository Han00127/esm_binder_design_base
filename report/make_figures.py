"""팀미팅 리포트용 figure 생성. 실제 실험 산출물(candidate JSON)에서 지표 계산 → PNG.

지표: 방향족(W+Y+F) 분율, 위치별 자연성 NLL(OAS q_target 기준), avg_ipsae(랭킹).
실험: baseline_proxy(no comp) / faithful(no comp) / KL-comp(30) [+ CE-comp 도착 시].
"""
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm        # 한글 폰트(Noto Sans CJK KR)
for _f in glob.glob("/usr/share/fonts/google-noto-cjk/NotoSansCJK-*.ttc"):
    try:
        _fm.fontManager.addfont(_f)
    except Exception:
        pass
# (영어 라벨 사용)
plt.rcParams["axes.unicode_minus"] = False

ESM = "/home/kyeongtak/structure_projects/esm_binder_design_base"
sys.path.insert(0, ESM)
os.chdir(ESM)

# ── 자연성 NLL: composition q_target(OAS) 기준 ──
from composition import CompositionTarget, _j_to_letter
ct = CompositionTarget("data/trastuzumab_qtarget_oas.npz", "cpu")
L2J = {l: j for j, l in enumerate(_j_to_letter())}
qpos = ct.q_pos.numpy()                       # [58,20] j-순서


def cdr_arom(cdr):
    return sum(c in "WYF" for c in cdr) / max(1, len(cdr))


def cdr_natnll(cdr):
    if len(cdr) != ct.n:
        return None
    v = [-np.log(qpos[i, L2J[c]]) for i, c in enumerate(cdr) if c in L2J]
    return float(np.mean(v)) if v else None


def load(pat):
    cds = []
    for f in glob.glob(pat):
        for c in json.load(open(f)).get("candidates", []):
            cds.append(c["cdr"])
    return cds


datasets = {}
for label, pat in [("baseline\n(no comp)", "runs/baseline_proxy/cand_g*.json"),
                   ("faithful\n(no comp)", "runs/cand_g*.json"),
                   ("KL-comp\n(30)", "runs/batch30_kl.json"),
                   ("CE-comp\n(30)", "runs/batch_ce_g*.json")]:
    cds = load(pat)
    if cds:
        datasets[label] = {"arom": [cdr_arom(c) for c in cds],
                           "nat": [n for c in cds if (n := cdr_natnll(c)) is not None],
                           "cdr": cds}

# native(trastuzumab) 기준선
NATIVE = "NIKDTYIHIYPTNGYTRYADWGGDGFYAMDYRASQDVNTAVASASFLYSQQHYTTPPT"  # 추출 native(58)
nat_native = cdr_natnll(NATIVE)
arom_native = cdr_arom(NATIVE)
NAT_DIST = 0.195   # OAS q_global 방향족
print("native nat_nll =", round(nat_native, 3), "arom =", round(arom_native, 3))
for k, v in datasets.items():
    print(f"{k.strip()}: arom {np.mean(v['arom']):.3f}  nat_nll {np.mean(v['nat']):.3f}  (n={len(v['cdr'])})")

labels = list(datasets)
colors = ["#bdbdbd", "#9e9e9e", "#4caf50", "#2196f3"][:len(labels)]

# ── Fig 1: 방향족 분율 ──
fig, ax = plt.subplots(figsize=(7, 4.2))
data = [datasets[l]["arom"] for l in labels]
bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
for p, c in zip(bp["boxes"], colors):
    p.set_facecolor(c); p.set_alpha(0.7)
ax.axhline(NAT_DIST, ls="--", c="red", label=f"natural Ab ~{NAT_DIST}")
ax.set_ylabel("CDR aromatic (W+Y+F) fraction")
ax.set_title("Aromatic fraction: composition prior pulls to natural level")
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig("report/fig1_aromatic.png", dpi=130); plt.close()

# ── Fig 2: 자연성 NLL ──
fig, ax = plt.subplots(figsize=(7, 4.2))
data = [datasets[l]["nat"] for l in labels]
bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
for p, c in zip(bp["boxes"], colors):
    p.set_facecolor(c); p.set_alpha(0.7)
ax.axhline(nat_native, ls="--", c="purple", label=f"native trastuzumab ({nat_native:.2f})")
ax.set_ylabel("position-wise naturalness NLL (lower = more natural)")
ax.set_title("CDR naturalness (vs OAS profile): comp << native (very germline-like)")
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig("report/fig2_naturalness.png", dpi=130); plt.close()

# ── Fig 3: avg_ipsae (랭킹; 있는 것만) ──
ipsae = {"baseline\n(no comp)": 0.145, "faithful\n(no comp)": 0.122}   # 이전 결과(top avg)
for label, rk in [("KL-comp\n(30)", "runs/rank_out/ranked.json"),
                  ("CE-comp\n(30)", "runs/rank_ce_out/ranked.json")]:
    if os.path.exists(rk):
        rows = json.load(open(rk))
        if rows:
            ipsae[label] = max(r.get("avg_ipsae", 0) for r in rows)
fig, ax = plt.subplots(figsize=(7, 4.2))
ks = list(ipsae); vs = [ipsae[k] for k in ks]
ax.bar(ks, vs, color=colors[:len(ks)], alpha=0.8)
for i, v in enumerate(vs):
    ax.text(i, v + 0.003, f"{v:.3f}", ha="center")
ax.set_ylabel("best avg ipSAE (4-critic)")
ax.set_title("4-critic ensemble ipSAE  (KEY: naturalness->transfer? comp ranking in progress)")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig("report/fig3_ipsae.png", dpi=130); plt.close()

# ── Fig 4: 예시 CDR 비교 (정성) ──
fig, ax = plt.subplots(figsize=(10, 3.2)); ax.axis("off")
ex = []
if "baseline\n(no comp)" in datasets:
    ex.append(("no-comp (baseline)", datasets["baseline\n(no comp)"]["cdr"][0], "#9e9e9e"))
if "KL-comp\n(30)" in datasets:
    ex.append(("KL-comp", datasets["KL-comp\n(30)"]["cdr"][0], "#4caf50"))
ex.append(("native trastuzumab", NATIVE, "#7e57c2"))
y = 0.8
ax.text(0.5, 0.97, "CDR sequence comparison (H1.H2.H3.L1.L2.L3 concat, 58 res)",
        ha="center", fontsize=12, weight="bold", transform=ax.transAxes)
for name, seq, c in ex:
    ax.text(0.01, y, f"{name:18s}", fontsize=10, family="monospace", weight="bold",
            color=c, transform=ax.transAxes)
    ax.text(0.27, y, seq, fontsize=9.5, family="monospace", transform=ax.transAxes)
    ax.text(0.27, y - 0.07, f"arom={cdr_arom(seq):.2f}  nat_nll={cdr_natnll(seq):.2f}",
            fontsize=8, color="gray", transform=ax.transAxes)
    y -= 0.27
plt.tight_layout(); plt.savefig("report/fig4_cdr_examples.png", dpi=130); plt.close()

# ── Fig 5: q_target.npz 내용 = 위치별 자연 분포(PSSM) 히트맵 ──
d = np.load("data/trastuzumab_qtarget_oas.npz", allow_pickle=True)
qp = d["q_pos"]                      # [58,20] 위치별 자연 분포 q
aa_order = str(d["aa_order"])
cdr_names = [str(x) for x in d["cdr_names"]]
fig, ax = plt.subplots(figsize=(13, 4.8))
im = ax.imshow(qp.T, aspect="auto", cmap="viridis")        # [20 AA, 58 pos]
ax.set_yticks(range(20)); ax.set_yticklabels(list(aa_order), fontsize=8)
ax.set_xlabel("CDR design position (58)"); ax.set_ylabel("amino acid")
ax.set_title("q_target.npz : natural CDR distribution per position (PSSM, from 64,523 Abs)")
prev = None
for i, name in enumerate(cdr_names):
    if name != prev:
        ax.axvline(i - 0.5, color="white", lw=1.2)
        ax.text(i + 0.3, -1.3, name, fontsize=9, color="black", weight="bold")
        prev = name
fig.colorbar(im, label="q = P(AA | position)")
plt.tight_layout(); plt.savefig("report/fig5_pssm_heatmap.png", dpi=130); plt.close()

# ── Fig 6: composition loss 동작 = CE(p,q), 보존 vs 가변 위치 예시 ──
ent = -(qp * np.log(qp + 1e-9)).sum(1)                     # 위치별 엔트로피
pos_cons, pos_div = int(ent.argmin()), int(ent.argmax())
nocomp = datasets["baseline\n(no comp)"]["cdr"][0]
comp = datasets["KL-comp\n(30)"]["cdr"][0]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
for ax, pos, tt in [(axes[0], pos_cons, "conserved position (low entropy)"),
                    (axes[1], pos_div, "diverse position (high entropy)")]:
    q = qp[pos]; order = np.argsort(-q); aas = [aa_order[j] for j in order]
    ax.bar(range(20), q[order], color="#90caf9")
    ax.set_xticks(range(20)); ax.set_xticklabels(aas, fontsize=8)
    ax.set_ylabel("q = natural P(AA)"); ax.set_title(f"{cdr_names[pos]}  pos#{pos}  — {tt}")
    yk = max(q)
    for cdr, nm, c, dy in [(NATIVE, "native", "#7e57c2", 0.92),
                           (nocomp, "no-comp", "#616161", 0.72),
                           (comp, "comp", "#2e7d32", 0.52)]:
        aa = cdr[pos]; rk = aas.index(aa) if aa in aas else 0
        ce = -np.log(q[L2J[aa]] + 1e-9)
        ax.annotate(f"{nm}={aa}  (-log q={ce:.1f})", xy=(rk, q[order][rk]),
                    xytext=(6, yk * dy), fontsize=8, color=c,
                    arrowprops=dict(arrowstyle="->", color=c, lw=1.2))
fig.suptitle("composition loss = CE(p,q) = -sum p*log q   |   pick high-q AA -> low loss (comp),  "
             "low-q AA -> high loss (no-comp)", fontsize=11)
plt.tight_layout(); plt.savefig("report/fig6_loss_mechanism.png", dpi=130); plt.close()

print("\n저장: report/fig1~6 (aromatic, naturalness, ipsae, cdr_examples, pssm_heatmap, loss_mechanism)")
