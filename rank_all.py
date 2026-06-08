"""rank_all.py — 병렬 생성된 후보들을 통합해 top-K 를 4-critic ipSAE ensemble 로 랭킹.

1) 후보 JSON(glob) 로드 → disto_iptm 상위 top-K 선별
2) Phase1: 후보별 첫 critic 폴딩(GPU 풀) → MSA 캐시 populate
3) Phase2: 나머지 critic 폴딩(캐시 hit, GPU 풀)
4) metrics 로 인터페이스 ipSAE → critic 평균 → 랭킹
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import time
from pathlib import Path

import yaml

import metrics
from critics import CRITICS, critic_path

ESM = "/home/kyeongtak/structure_projects/esm_binder_design_base"
PY = "/home/kyeongtak/.conda/envs/esmfold2/bin/python"


def write_yaml(out_dir, cand, ag_id, ag_seq, ab="S"):
    yml = f"{out_dir}/{cand['name']}.yaml"
    if not Path(yml).exists():
        yaml.dump({"chains": [{"type": "protein", "id": ag_id, "sequence": ag_seq},
                              {"type": "protein", "id": ab, "sequence": cand["scfv"]}],
                   "inference": {"num_loops": 10, "num_sampling_steps": 64,
                                 "num_diffusion_samples": 1}}, open(yml, "w"), sort_keys=False)
    return yml


def launch(out_dir, cand, ck, msa, gpu):
    yml = f"{out_dir}/{cand['name']}.yaml"
    stem = f"{out_dir}/{cand['name']}_{ck}"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "HF_HUB_OFFLINE": "1",
           "TRANSFORMERS_OFFLINE": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    log = open(stem + ".log", "w")
    return subprocess.Popen(
        [PY, f"{ESM}/run_esmfold2.py", "--input", yml, "--output", stem + ".cif",
         "--msa", msa, "--weights", critic_path(ck)],
        stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ESM)


def run_pool(jobs, gpus):
    """jobs: list of (cand, ck, out_dir, msa). GPU 풀로 병렬 실행."""
    free = list(gpus); running = {}; i = 0
    while i < len(jobs) or running:
        while free and i < len(jobs):
            g = free.pop(0); cand, ck, out_dir, msa = jobs[i]; i += 1
            running[g] = (launch(out_dir, cand, ck, msa, g), f"{cand['name']}_{ck}")
        time.sleep(8)
        for g, (p, nm) in list(running.items()):
            if p.poll() is not None:
                del running[g]; free.append(g)


def score(out_dir, cand, ck, ag_id, ab="S"):
    stem = f"{out_dir}/{cand['name']}_{ck}"
    if not Path(stem + ".cif").exists():
        return None
    try:
        ev = metrics.evaluate_complex(stem + ".cif", stem + "_pae.npy", stem + "_confidence.json")
        p = ev["pairs"].get((ab, ag_id)) or ev["pairs"].get((ag_id, ab)) or {}
        return p.get("ipsae_max", 0.0)
    except Exception as e:
        print("  metrics err", e); return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", default=f"{ESM}/runs/cand_g*.json")
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--msa", default="auto")
    ap.add_argument("--gpus", default="0,1,2,3,4,5")
    ap.add_argument("--out-dir", default=f"{ESM}/runs/rank_out")
    args = ap.parse_args()
    gpus = [int(g) for g in args.gpus.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)

    cands, ag_id, ag_seq = [], None, None
    for f in sorted(glob.glob(args.cands)):
        d = json.load(open(f)); ag_id, ag_seq = d["antigen_id"], d["antigen_seq"]
        cands += d["candidates"]
    print(f"[rank_all] 후보 {len(cands)}개 로드 → disto_iptm 상위 {args.topk} 선별")
    cands.sort(key=lambda c: -c.get("disto_iptm", 0))
    top = cands[:args.topk]
    for c in top:
        write_yaml(args.out_dir, c, ag_id, ag_seq)

    cks = list(CRITICS)
    # Phase1: 후보별 첫 critic (MSA populate)
    print(f"[rank_all] Phase1: MSA populate ({len(top)} 후보 × {cks[0]})")
    run_pool([(c, cks[0], args.out_dir, args.msa) for c in top], gpus)
    # Phase2: 나머지 critic
    print(f"[rank_all] Phase2: 나머지 critic {cks[1:]} (캐시 hit)")
    run_pool([(c, ck, args.out_dir, args.msa) for c in top for ck in cks[1:]], gpus)

    # 채점 + 랭킹
    rows = []
    for c in top:
        per = {ck: score(args.out_dir, c, ck, ag_id) for ck in cks}
        vals = [v for v in per.values() if v is not None]
        avg = sum(vals) / len(vals) if vals else 0.0
        rows.append({**c, "per_critic": per, "avg_ipsae": round(avg, 4)})
    rows.sort(key=lambda r: -r["avg_ipsae"])
    json.dump(rows, open(f"{args.out_dir}/ranked.json", "w"), indent=2)
    print("\n[rank_all] ★ RANKED (avg_ipsae 내림차순):")
    for r in rows:
        pc = " ".join(f"{k}={('%.3f'%v) if v is not None else 'NA'}" for k, v in r["per_critic"].items())
        print(f"  {r['name']}: avg={r['avg_ipsae']} (disto={r.get('disto_iptm')}) [{pc}] CDR={r['cdr']}")


if __name__ == "__main__":
    main()
