"""rank_completed_fv.py — 완료된 H3-fix 설계를 *Fv 형태 · multi-seed×sample* 로 full STEAP1 trimer 에
fold(--msa auto)해 binder↔protomer ipSAE 로 랭킹. scFv 는 floating(ipSAE 0)이라 rank.py(scFv) 대신 이걸 쓴다.

  설계 VH = scfv[:len(vh)] (H3 만 부분설계됨) + native VL → trimer A/B/C + H + L 복합체.
  설계당 fold_multiseed.py 1회(모델·MSA 1회 로드 → N_SEED seed × N_SAMPLE diffusion samples = 구조 N개).
  구조별 binder↔protomer best ipSAE → **평균·max 집계**로 랭킹.
  (벤치: loops5·samples3·seed5 = 15구조/설계, loops10·seed10 대비 3.4× 빠름. diffusion sample 은 trunk
   재사용이라 거의 공짜; s5 는 L=1283서 OOM → s3 한계.)

사용:  PYTHONPATH=. python rank_completed_fv.py [designs.json] [gpu] [n_seed]
  idempotent: fold_multiseed --skip-existing → 이미 fold된 (설계,seed,sample)은 건너뜀. 누적/재개/증설 가능.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

import metrics

ESM = "/home/kyeongtak/structure_projects/esm_binder_design_base"
PY = "/home/kyeongtak/.conda/envs/esmfold2/bin/python"
WEIGHTS = "/home/aidx/DB/weights/esmfold2/ESMFold2"
AG_IDS, AB_IDS = ("A", "B", "C"), ("H", "L")
N_SEED_DEFAULT = 5          # seed 수 (독립 trunk pass)
N_SAMPLE = 3                # seed당 diffusion samples (trunk 재사용 거의 공짜; s5 OOM)
N_LOOPS = 5                 # recycle. 재baseline 검증 후 확정(분별력 깨지면 10으로)


def _fold_multiseed(yml, stem, gpu, n_seed, msa="auto"):
    """fold_multiseed.py 1회 = 모델·MSA 1회 로드 후 n_seed×N_SAMPLE fold. {stem}_seed{s}_sample{i}.* 저장."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "HF_HUB_OFFLINE": "1",
           "TRANSFORMERS_OFFLINE": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    cmd = [PY, f"{ESM}/fold_multiseed.py", "--input", yml, "--output-stem", stem,
           "--n-seed", str(n_seed), "--num-loops", str(N_LOOPS),
           "--num-diffusion-samples", str(N_SAMPLE),
           "--msa", msa, "--weights", WEIGHTS, "--skip-existing"]
    r = subprocess.run(cmd, env=env, cwd=ESM, capture_output=True, text=True,
                       timeout=n_seed * 600 + 1800)
    if not Path(f"{stem}_seed0_sample0.cif").exists():
        print("  fold 실패:", (r.stderr or "")[-300:])


def _binder_ipsae(stem):
    """binder↔protomer ipSAE(ab×ag max) per protomer. 없으면 None."""
    cif, pae, cj = stem + ".cif", stem + "_pae.npy", stem + "_confidence.json"
    if not Path(cif).exists():
        return None
    ev = metrics.evaluate_complex(cif, pae, cj)
    return {ag: max((ev["pairs"].get((ab, ag)) or ev["pairs"].get((ag, ab)) or {})
                    .get("ipsae_max", 0) for ab in AB_IDS) for ag in AG_IDS}


def main():
    inp = sys.argv[1] if len(sys.argv) > 1 else f"{ESM}/runs/h3fix_designs.json"
    gpu = sys.argv[2] if len(sys.argv) > 2 else "7"
    n_seed = int(sys.argv[3]) if len(sys.argv) > 3 else N_SEED_DEFAULT
    cands = json.load(open(inp))["candidates"]

    fv0 = yaml.safe_load(open(f"{ESM}/runs/foldfv_k50_seed0.yaml"))      # full STEAP1 트리머 재사용
    trimer = [{"type": "protein", "id": c["id"], "sequence": c["sequence"]}
              for c in fv0["chains"] if c["id"] in AG_IDS]
    cfg = yaml.safe_load(open(f"{ESM}/configs/full_steap1_trimer_fv.yaml"))
    _clean = lambda s: s.replace(" ", "").replace("\n", "")
    vh = _clean(cfg["antibody"]["heavy"]["vh_sequence"])
    vl = _clean(cfg["antibody"]["light"]["vl_sequence"])

    _base = os.path.basename(inp).replace("_designs.json", "").replace(".json", "")
    out_dir = f"{ESM}/runs/{_base}_rank_fv"          # 입력별 분리(h3fix/lmap 안 섞임)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[rank-fv] {len(cands)} 설계 × {n_seed}seed×{N_SAMPLE}sample (loops={N_LOOPS}) → "
          f"Fv fold(--msa auto, GPU{gpu})  baseline: ref(native) mean=0.17/max=0.23 / 기존 full설계=0.00\n")
    rows = []
    for c in cands:
        des_vh = c["scfv"][:len(vh)]                                     # 설계 H3 포함 VH
        chains = trimer + [{"type": "protein", "id": "H", "sequence": des_vh},
                           {"type": "protein", "id": "L", "sequence": vl}]
        stem0 = f"{out_dir}/{c['name']}"
        yml = f"{stem0}.yaml"
        yaml.dump({"chains": chains, "inference": {"num_loops": N_LOOPS, "num_sampling_steps": 64,
                   "num_diffusion_samples": N_SAMPLE}}, open(yml, "w"), sort_keys=False)
        stems = [f"{stem0}_seed{s}_sample{i}" for s in range(n_seed) for i in range(N_SAMPLE)]
        if any(not Path(st + ".cif").exists() for st in stems):
            _fold_multiseed(yml, stem0, gpu, n_seed)                     # 설계당 1회(모델 1회 로드)

        per_struct_best, per_prot = [], {ag: [] for ag in AG_IDS}
        for st in stems:
            ips = _binder_ipsae(st)
            if ips is None:
                continue
            per_struct_best.append(max(ips.values()))
            for ag in AG_IDS:
                per_prot[ag].append(ips[ag])
        if not per_struct_best:
            print(f"[rank-fv] {c['name']}: (fold 결과 없음)")
            continue
        mean_ip = round(float(np.mean(per_struct_best)), 3)
        max_ip = round(float(np.max(per_struct_best)), 3)
        prot_mean = {ag: round(float(np.mean(per_prot[ag])), 3) for ag in AG_IDS}   # quaternary 진단
        rows.append({"name": c["name"], "cdr": c.get("cdr"), "disto_iptm": c.get("disto_iptm"),
                     "n_struct": len(per_struct_best), "mean_ipsae": mean_ip, "max_ipsae": max_ip,
                     "prot_mean_ipsae": prot_mean})
        print(f"[rank-fv] {c['name']}: mean={mean_ip} max={max_ip} (n={len(per_struct_best)})  "
              f"protomer평균(A,B,C)={prot_mean}  CDR={c.get('cdr')}")

    rows.sort(key=lambda r: -r["mean_ipsae"])                           # 평균 기준 랭킹
    json.dump(rows, open(f"{out_dir}/ranked_fv.json", "w"), indent=2)
    print("\n★ RANKED (구조 평균 ipSAE 내림차순; max 병기):")
    for r in rows:
        print(f"  {r['name']}: mean={r['mean_ipsae']} max={r['max_ipsae']} "
              f"protomer평균={r['prot_mean_ipsae']}  CDR={r['cdr']}")


if __name__ == "__main__":
    main()
