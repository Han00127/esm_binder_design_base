"""native_sweep.py — native 항체(설계 X)의 L_inter 를 num_loops(1/3/5/10)별로,
H3-loop vs 다른CDR 분리 측정. "distogram이 도킹을 표현하나 + num_loops 효과 + 66쌍 중 어디가 문제냐"를 한 번에.

forward-only(no_grad)라 단일 GPU 에 들어감 (pipeline 불필요).
사용:  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python native_sweep.py
"""
from __future__ import annotations

import sys

import numpy as np
import torch
import yaml

import losses
import scfv as scfvmod
from esm.models.esmfold2 import (ESMFold2InputBuilder, ProteinInput,
                                 StructurePredictionInput)
from esmfold_diff import distogram_forward, load

AA_BASE = 2
LINKER = "GGGGSGGGGSGGGGS"
DEV = "cuda:0"
_cl = lambda s: s.replace(" ", "").replace("\n", "")

cfg = yaml.safe_load(open("configs/full_steap1_trimer_fv.yaml"))
vh = _cl(cfg["antibody"]["heavy"]["vh_sequence"])
vl = _cl(cfg["antibody"]["light"]["vl_sequence"])
H3OVR = sys.argv[2] if len(sys.argv) > 2 else None      # graft: H3(96-110) 14자 교체
if H3OVR:
    assert len(H3OVR) == 14, "H3 override 는 14자(96-109)"
    vh = vh[:96] + H3OVR + vh[110:]
    print(f"[graft] H3 → {H3OVR}  (native=TRWGYYGTRGYFNV)")
scfv = scfvmod.make_scfv(vh, vl, LINKER, "VH-VL")["seq"]
ag_chains = [(c["id"], _cl(c["sequence"])) for c in cfg["chains"]]

builder = ESMFold2InputBuilder()
chains = [ProteinInput(id=cid, sequence=s) for cid, s in ag_chains] + \
         [ProteinInput(id="S", sequence=scfv)]
feats, _ = builder.prepare_input(StructurePredictionInput(sequences=chains), device=DEV)
rt = feats["res_type"]
L = rt.shape[1]
print(f"[native_sweep] L={L} | native H3 = {scfv[96:110]}")

# native one-hot soft (전 위치 native AA)
prompt_ids = [(int(i) - AA_BASE) if (AA_BASE <= int(i) <= AA_BASE + 19) else None
              for i in rt[0].tolist()]
soft = torch.zeros(L, 20, device=DEV)
for i, j in enumerate(prompt_ids):
    if j is not None:
        soft[i, j] = 1.0
soft_full = torch.zeros(1, L, 33, device=DEV)
soft_full[0, :, AA_BASE:AA_BASE + 20] = soft

model, raw_fwd = load(device=DEV)

z = np.load("data/lmap_targets.npz", allow_pickle=True)
b = torch.as_tensor(z["inter_b_idx"], dtype=torch.long, device=DEV)
a = torch.as_tensor(z["inter_a_idx"], dtype=torch.long, device=DEV)
t = torch.as_tensor(z["inter_target_caca"], dtype=torch.float32, device=DEV)
h3 = torch.as_tensor([str(r).startswith("H3") for r in z["inter_region"]], device=DEV)
print(f"[native_sweep] 접촉 {len(b)}쌍 = H3-loop {int(h3.sum())} + 다른CDR {int((~h3).sum())}\n")


# ── CDR 라벨 + 설계가능(mutable H3 102-109) 마스크 (run.py 와 동일 규칙) ──
sl = z["inter_b_idx"].astype(int) - 759


def _cdr(s):
    if 25 <= s <= 34: return "H1"
    if 46 <= s <= 65: return "H2"
    if 96 <= s <= 109: return "H3"
    if 159 <= s <= 170: return "L1"
    if 181 <= s <= 191: return "L2"
    if 224 <= s <= 234: return "L3"
    return "FR"


labels = [_cdr(int(s)) for s in sl]
cdr_masks = {c: torch.as_tensor([lb == c for lb in labels], device=DEV)
             for c in ("H1", "H2", "H3", "L1", "L2", "L3")}
design_mask = torch.as_tensor([861 <= int(bb) <= 868 for bb in z["inter_b_idx"]], device=DEV)  # 설계 9쌍


def rms(d, m):
    if not bool(m.any()):
        return None
    return round(5 * float(losses.map_inter(d, b[m], a[m], t[m])) ** 0.5, 1)


COLS = [("설계9", design_mask)] + [(c, cdr_masks[c]) for c in ("H1", "H2", "H3", "L1", "L2", "L3")]
LOOPS = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [1, 3, 5, 10]
print("native 접촉 RMS(Å) — 그룹별 (= 설계가 도달할 천장)")
print("loops | " + " ".join(f"{c:>5}" for c, _ in COLS))
print("-" * 60)
for nl in LOOPS:
    with torch.no_grad():
        d = distogram_forward(model, raw_fwd, feats, soft_full, num_loops=nl)
    vals = [rms(d, m) for _, m in COLS]
    print(f"{nl:>5} | " + " ".join(f"{(v if v is not None else 0):>5.1f}" for v in vals), flush=True)
print("\n(설계9 = mutable H3 102·103·106·108 접촉 = wandb rms_design 의 천장)")
