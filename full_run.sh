#!/bin/bash
# full_run.sh [TRAJ_PER_GPU] [K] [TOPK]
# Phase1: GPU 0-5 병렬로 각 TRAJ개 trajectory 생성(Alg11, K step) → cand_gX.json
# Phase2: 전체 후보 통합 → disto_iptm 상위 TOPK → 4-critic msa=auto 랭킹
set -u
cd /home/kyeongtak/structure_projects/esm_binder_design_base
mkdir -p runs
T=${1:-2}; K=${2:-150}; TOPK=${3:-6}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/kyeongtak/.conda/envs/esmfold2/bin/python

echo "[full] Phase1: GPU 0-5 병렬 생성 (각 $T traj × K=$K) $(date)"
for G in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$G $PY run.py --trajectories $T --steps $K \
    --seed-base $((G*100)) --out runs/cand_g$G.json > runs/gen_g$G.log 2>&1 &
done
wait
echo "[full] 생성 완료 $(date) — 후보 파일:"
ls runs/cand_g*.json 2>/dev/null

echo "[full] Phase2: 통합 top-$TOPK 4-critic 랭킹 (msa=auto) $(date)"
$PY rank_all.py --cands "runs/cand_g*.json" --topk $TOPK --msa auto --gpus 0,1,2,3,4,5 \
  > runs/rank_all.log 2>&1
echo "[full] DONE $(date)"
tail -20 runs/rank_all.log
