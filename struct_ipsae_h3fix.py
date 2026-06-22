"""struct_ipsae_h3fix.py — h3fix_rank_fv 의 모든 fold(설계×seed×sample) + reference(native) 를
8UCD에 정렬해 *구조기반 PAE(AE)* 로 ipSAE 계산. 모델 자기보고 PAE(ranked_fv.json)와 비교.
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np

from validate_vs_8ucd import struct_based

OUT = "runs/_h3fix_struct"
os.makedirs(OUT, exist_ok=True)

groups = defaultdict(list)
for cif in sorted(glob.glob("runs/h3fix_rank_fv/*_seed*_sample*.cif")):
    name = os.path.basename(cif)[:-4].split("_seed")[0]
    groups[name].append(cif)
refs = sorted(glob.glob("runs/_rebase/ref_seed*_sample*.cif"))
if refs:
    groups["REF_native"] = refs

mp = {}
try:
    for r in json.load(open("runs/h3fix_rank_fv/ranked_fv.json")):
        mp[r["name"]] = (r["mean_ipsae"], r["max_ipsae"])
except Exception:
    pass

rows = []
for name, cifs in groups.items():
    bests, rmsds = [], []
    for cif in cifs:
        stem = os.path.basename(cif)[:-4]
        try:
            bind, rmsd, cov = struct_based(cif, f"{OUT}/{stem}_spae.npy", f"{OUT}/{stem}_sconf.json")
            bests.append(max(bind.values())); rmsds.append(rmsd)
        except Exception as e:
            print(f"  err {stem}: {str(e)[:50]}")
    if not bests:
        continue
    m, mx, rmed = float(np.mean(bests)), float(np.max(bests)), float(np.median(rmsds))
    mpm = mp.get(name, (None, None))
    rows.append({"name": name, "struct_mean": round(m, 3), "struct_max": round(mx, 3),
                 "rmsd_med": round(rmed, 1), "n": len(bests),
                 "model_mean": mpm[0], "model_max": mpm[1]})
    print(f"[{name}] 구조기반 mean={m:.3f} max={mx:.3f} RMSD중앙={rmed:.1f}Å (n={len(bests)}) "
          f"| 모델PAE mean={mpm[0]}", flush=True)

rows.sort(key=lambda r: -r["struct_mean"])
json.dump(rows, open(f"{OUT}/struct_ranked.json", "w"), indent=2)
print("\n★ 구조기반(vs8UCD) ipSAE 랭킹  [모델PAE와 비교]")
print(f"{'설계':12} {'구조mean':>8} {'구조max':>8} {'RMSD':>6} {'모델PAE_mean':>12} {'모델PAE_max':>11}")
for r in rows:
    print(f"{r['name']:12} {r['struct_mean']:>8.3f} {r['struct_max']:>8.3f} {r['rmsd_med']:>6.1f} "
          f"{(r['model_mean'] or 0):>12.3f} {(r['model_max'] or 0):>11.3f}")
print("\n해석: 구조기반=실제 8UCD 재현 | 모델PAE=모델 자기확신. 둘 차이 = confident-but-wrong 여부.")
