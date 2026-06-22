"""fold_multiseed.py — 모델·MSA를 한 번만 로드하고 여러 seed(× diffusion samples)로 반복 fold.

run_esmfold2.py 는 invocation 마다 (CUDA init + 모델 weight 로드 ~2-3분)를 반복 → seed 10개면 그 고정비용
10배. 이 wrapper 는 ① _build_spi(MSA 캐시 1회) ② 모델 로드 1회 ③ seed 루프 fold+저장 → 고정비용 1배.
num_diffusion_samples>1 이면 1회 fold(=trunk 1회)로 여러 구조 → trunk 재사용으로 크게 절감.
저장 포맷(cif / _confidence.json / _pae.npy)은 run_esmfold2 와 동일 → metrics.evaluate_complex 호환.

사용:  CUDA_VISIBLE_DEVICES=7 python fold_multiseed.py --input X.yaml --output-stem DIR/name \
         --n-seed 5 [--num-diffusion-samples 5] [--num-loops 5] [--msa auto] [--skip-existing]
  결과: {stem}_seed{s}.cif (samples=1) | {stem}_seed{s}_sample{i}.cif (samples>1) + _confidence.json/_pae.npy
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from run_esmfold2 import (DEFAULT_WEIGHTS, ESMFold2InputBuilder, ESMFold2Model,
                          _build_spi)


def _save(result, stem: str):
    """run_esmfold2._save_result 와 동일 포맷(cif/confidence/pae)."""
    Path(stem).parent.mkdir(parents=True, exist_ok=True)
    plddt = result.plddt.float().cpu().numpy()
    conf = {"plddt_mean": round(float(plddt.mean()), 4),
            "ptm": round(float(result.ptm), 4) if result.ptm is not None else None,
            "iptm": round(float(result.iptm), 4) if result.iptm is not None else None,
            "plddt_per_token": [round(float(v), 4) for v in plddt]}
    if result.pair_chains_iptm is not None:
        conf["pair_chains_iptm"] = result.pair_chains_iptm.float().cpu().numpy().tolist()
    Path(stem + ".cif").write_text(result.complex.to_mmcif())
    Path(stem + "_confidence.json").write_text(json.dumps(conf, indent=2))
    if result.pae is not None:
        np.save(stem + "_pae.npy", result.pae.float().cpu().numpy())
    return conf["plddt_mean"], conf.get("iptm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", required=True)
    ap.add_argument("--output-stem", "-o", required=True, help="저장 베이스; {stem}_seed{s}[_sample{i}].* 저장")
    ap.add_argument("--weights", "-w", default=DEFAULT_WEIGHTS)
    ap.add_argument("--msa", default="auto", choices=["yaml", "auto", "none"])
    ap.add_argument("--msa-select", type=int, default=None)
    ap.add_argument("--n-seed", type=int, default=5)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--num-loops", type=int, default=5)
    ap.add_argument("--num-sampling-steps", type=int, default=64)
    ap.add_argument("--num-diffusion-samples", type=int, default=3)  # s5는 L=1283서 OOM → s3 한계
    ap.add_argument("--skip-existing", action="store_true",
                    help="이미 있는 seed cif 는 건너뜀(idempotent; 재개/seed증설용)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.input))
    chains = cfg["chains"]

    # ① 입력+MSA 1회 (sha256 캐시 hit → search 없음)
    sel = {"select": args.msa_select} if args.msa_select is not None else {}
    spi = _build_spi(chains, Path(args.input).parent, msa_mode=args.msa, **sel)

    # ② 모델 1회 로드 (CUDA init + weight 로드)
    t0 = time.time()
    print(f"[multiseed] load model {args.weights}")
    model = ESMFold2Model.from_pretrained(args.weights, local_files_only=True).cuda().eval()
    builder = ESMFold2InputBuilder()
    print(f"[multiseed] model load {time.time()-t0:.0f}s | "
          f"loops={args.num_loops} steps={args.num_sampling_steps} samples={args.num_diffusion_samples}")

    # ③ seed 루프 (모델·spi 재사용 → fold 연산만 반복)
    ns = args.num_diffusion_samples
    seeds = list(range(args.seed_start, args.seed_start + args.n_seed))
    for s in seeds:
        base = f"{args.output_stem}_seed{s}"
        if args.skip_existing and Path((base if ns == 1 else base + "_sample0") + ".cif").exists():
            print(f"[multiseed] seed{s}: skip (이미 존재)")
            continue
        tf = time.time()
        raw = builder.fold(model, spi, num_loops=args.num_loops,
                           num_sampling_steps=args.num_sampling_steps,
                           num_diffusion_samples=ns, seed=s)
        results = raw if isinstance(raw, list) else [raw]
        for i, result in enumerate(results):
            stem = base if ns == 1 else f"{base}_sample{i}"
            pl, ip = _save(result, stem)
            print(f"[multiseed] seed{s} sample{i}: pLDDT={pl:.3f} ipTM={ip}")
        print(f"[multiseed] seed{s} DONE {time.time()-tf:.0f}s ({len(results)} 구조)")


if __name__ == "__main__":
    main()
