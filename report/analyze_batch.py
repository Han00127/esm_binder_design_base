"""배치 종합 분석: 시퀀스(조성·자연성·다양성) + 구조(plddt) + 4-critic ipSAE.
사용 가능한 배치(baseline/KL-comp/CE-comp) 자동 분석 + 비교 figure. 재실행 가능(idempotent).
"""
import glob
import itertools
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["axes.unicode_minus"] = False
import sys
ESM = "/home/kyeongtak/structure_projects/esm_binder_design_base"
sys.path.insert(0, ESM)
os.chdir(ESM)
import yaml
from composition import CompositionTarget, _j_to_letter

ct = CompositionTarget("data/trastuzumab_qtarget_oas.npz", "cpu")
L2J = {l: j for j, l in enumerate(_j_to_letter())}
qpos = ct.q_pos.numpy()

cfg = yaml.safe_load(open("configs/trastuzumab_her2.yaml"))
segs, off = [], 0
for ch in ("heavy", "light"):
    for n, (s, e) in cfg["antibody"][ch]["cdr_ranges"].items():
        segs.append((("H" if ch == "heavy" else "L") + n[-1], off, off + (e - s))); off += (e - s)


def arom(c):
    return sum(x in "WYF" for x in c) / max(1, len(c))


def natnll(c):
    if len(c) != ct.n:
        return None
    return float(np.mean([-np.log(qpos[i, L2J[a]]) for i, a in enumerate(c) if a in L2J]))


BATCHES = [
    ("baseline", "runs/baseline_proxy/rank_out/ranked.json", "runs/baseline_proxy/rank_out", "#9e9e9e"),
    ("KL-comp",  "runs/rank_out/ranked.json",                "runs/rank_out",                "#2e7d32"),
    ("CE-comp",  "runs/rank_ce_out/ranked.json",             "runs/rank_ce_out",             "#1565c0"),
]


def load(ranked, rdir):
    if not os.path.exists(ranked):
        return None
    rows = json.load(open(ranked))
    if not rows:
        return None
    for r in rows:
        c = r["cdr"]
        r["arom"] = arom(c); r["natnll"] = natnll(c)
        pl = []
        for f in glob.glob(f"{rdir}/{r['name']}_*_confidence.json"):
            try:
                v = json.load(open(f)).get("plddt_mean")
                if v:
                    pl.append(v if v > 1 else v * 100)
            except Exception:
                pass
        r["plddt"] = float(np.mean(pl)) if pl else None
    return rows


def diversity(cdrs):
    ids = [sum(x == y for x, y in zip(a, b)) / len(a)
           for a, b in itertools.combinations(cdrs, 2) if len(a) == len(b)]
    return np.array(ids) if ids else np.array([np.nan])


data = {}
for name, rk, rdir, col in BATCHES:
    rows = load(rk, rdir)
    if rows:
        data[name] = dict(rows=rows, col=col,
                          ipsae=[r["avg_ipsae"] for r in rows],
                          arom=[r["arom"] for r in rows],
                          nat=[r["natnll"] for r in rows if r["natnll"] is not None],
                          plddt=[r["plddt"] for r in rows if r["plddt"]],
                          div=diversity([r["cdr"] for r in rows]))
        print(f"[{name}] n={len(rows)}  best_ipsae={max(d for d in [r['avg_ipsae'] for r in rows]):.3f}  "
              f"mean_ipsae={np.mean([r['avg_ipsae'] for r in rows]):.3f}  "
              f"arom={np.mean([r['arom'] for r in rows]):.3f}  "
              f"natnll={np.mean([r['natnll'] for r in rows if r['natnll'] is not None]):.2f}  "
              f"plddt={np.mean([r['plddt'] for r in rows if r['plddt']]) if any(r['plddt'] for r in rows) else float('nan'):.1f}  "
              f"div_id={np.nanmean(diversity([r['cdr'] for r in rows])):.2f}")

if not data:
    print("분석할 ranked.json 없음"); raise SystemExit
labels = list(data); cols = [data[k]["col"] for k in labels]


def boxpanel(ax, key, title, ylabel, hline=None):
    vals = [data[k][key] for k in labels]
    bp = ax.boxplot(vals, labels=labels, patch_artist=True, widths=0.5)
    for p, k in zip(bp["boxes"], labels):
        p.set_facecolor(data[k]["col"]); p.set_alpha(0.6)
    if hline is not None:
        ax.axhline(hline, ls="--", c="red", lw=1)
    ax.set_title(title); ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=0.3)


# ── Fig: 종합 비교 (ipSAE / 자연성 / 방향족 / 다양성 / plddt) ──
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
boxpanel(axes[0, 0], "ipsae", "4-critic avg ipSAE (higher=better)", "avg ipSAE")
boxpanel(axes[0, 1], "nat", "CDR naturalness NLL (lower=natural)", "nat NLL", hline=2.03)
boxpanel(axes[0, 2], "arom", "aromatic (W+Y+F) fraction", "fraction", hline=0.195)
boxpanel(axes[1, 0], "plddt", "structure pLDDT (critic fold)", "pLDDT")
boxpanel(axes[1, 1], "div", "sequence diversity (pairwise identity; lower=diverse)", "identity")
# scatter: nat vs ipSAE
ax = axes[1, 2]
for k in labels:
    rr = data[k]["rows"]
    ax.scatter([r["natnll"] for r in rr if r["natnll"] is not None],
               [r["avg_ipsae"] for r in rr if r["natnll"] is not None],
               c=data[k]["col"], label=k, alpha=0.7, s=22)
ax.set_xlabel("naturalness NLL"); ax.set_ylabel("avg ipSAE")
ax.set_title("naturalness vs ipSAE (transfer?)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.suptitle("Batch comparison: sequence + structure + critic ipSAE", fontsize=14, weight="bold")
plt.tight_layout(); plt.savefig("report/figA_batch_compare.png", dpi=135); plt.close()

# ── Fig: per-critic ipSAE (배치별, 4 critic) ──
crit = sorted({c for k in labels for r in data[k]["rows"] for c in (r.get("per_critic") or {})})
if crit:
    fig, ax = plt.subplots(figsize=(11, 5))
    xw = 0.8 / len(labels)
    xc = np.arange(len(crit))
    for i, k in enumerate(labels):
        means = [np.mean([(r.get("per_critic") or {}).get(c) or 0 for r in data[k]["rows"]]) for c in crit]
        ax.bar(xc + i * xw, means, xw, label=k, color=data[k]["col"], alpha=0.8)
    ax.set_xticks(xc + xw * (len(labels) - 1) / 2); ax.set_xticklabels(crit, fontsize=8)
    ax.set_ylabel("mean ipSAE"); ax.set_title("per-critic mean ipSAE by batch")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig("report/figB_per_critic.png", dpi=135); plt.close()

print("\n저장: report/figA_batch_compare.png" + (", figB_per_critic.png" if crit else ""))
