"""rank.py — 후처리 4-critic ipTM/ipSAE ensemble 랭킹 (논문 후처리).

생성된 각 후보(scFv)를 [항원, scFv] 복합체로 critic 들로 폴딩(no-grad) → 인터페이스 confidence
(ipSAE) → critic 평균 = 랭킹 점수. 검증된 run_esmfold2(subprocess) + metrics 재사용.
critic = Experimental 4종(critics.py). (in-process 최적화는 추후; baseline 검증엔 subprocess.)
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import yaml

import metrics
from critics import CRITICS, critic_path

ESM = "/home/kyeongtak/structure_projects/esm_binder_design_base"
PY = "/home/kyeongtak/.conda/envs/esmfold2/bin/python"


def _fold(yaml_path, out_cif, weights, msa, gpu):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "HF_HUB_OFFLINE": "1",
           "TRANSFORMERS_OFFLINE": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    cmd = [PY, f"{ESM}/run_esmfold2.py", "--input", yaml_path, "--output", out_cif, "--msa", msa]
    if weights:
        cmd += ["--weights", weights]
    try:
        subprocess.run(cmd, env=env, cwd=ESM, capture_output=True, text=True, timeout=1800)
    except Exception as e:
        print("  fold err", e)


def rank(candidates, antigen_seq, antigen_id, out_dir, critic_keys=None,
         msa="auto", gpu=1, ab_chain="S"):
    """candidates: [{'name','scfv'}]. 반환: avg_ipsae 내림차순 정렬 rows."""
    critic_keys = critic_keys or list(CRITICS)
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for c in candidates:
        yml = f"{out_dir}/{c['name']}.yaml"
        yaml.dump({"chains": [{"type": "protein", "id": antigen_id, "sequence": antigen_seq},
                              {"type": "protein", "id": ab_chain, "sequence": c["scfv"]}],
                   "inference": {"num_loops": 10, "num_sampling_steps": 64,
                                 "num_diffusion_samples": 1}},
                  open(yml, "w"), sort_keys=False)
        per_critic = {}
        for ck in critic_keys:
            stem = f"{out_dir}/{c['name']}_{ck}"
            _fold(yml, stem + ".cif", critic_path(ck), msa, gpu)
            ip = None
            if Path(stem + ".cif").exists():
                try:
                    ev = metrics.evaluate_complex(stem + ".cif", stem + "_pae.npy",
                                                  stem + "_confidence.json")
                    p = (ev["pairs"].get((ab_chain, antigen_id))
                         or ev["pairs"].get((antigen_id, ab_chain)) or {})
                    ip = p.get("ipsae_max", 0.0)
                except Exception as e:
                    print("  metrics err", e)
            per_critic[ck] = ip
        vals = [v for v in per_critic.values() if v is not None]
        avg = sum(vals) / len(vals) if vals else 0.0
        rows.append({**c, "per_critic": per_critic, "avg_ipsae": round(avg, 4)})
        print(f"[rank] {c['name']}: avg_ipsae={avg:.4f}  "
              + " ".join(f"{k}={('%.3f'%v) if v is not None else 'NA'}" for k, v in per_critic.items()))
    rows.sort(key=lambda r: -r["avg_ipsae"])
    return rows
