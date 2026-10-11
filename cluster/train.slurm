#!/bin/bash
#SBATCH --job-name=c2g-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
# Train one ensemble member. From the repo root (logs/ must exist before submitting):
#     sbatch -J b5_coast_s0 cluster/train.slurm configs/b5_coast.yaml 0 runs/b5_coast_s0
# Extra arguments go to canopy2ground.train. --constraint=gpu80 asks for an 80 GB A100 (Princeton naming;
# drop or rename it elsewhere). An A100 trains a 66-epoch MiT-B2 member in ~1.5 h and MiT-B5 in ~2.7 h.
set -eo pipefail
CONFIG=${1:?config}; SEED=${2:?seed}; OUT=${3:?output dir}; shift 3
cd "${SLURM_SUBMIT_DIR:-.}"          # sbatch runs a spooled copy of this script
source cluster/env.sh
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
exec "$C2G_PYTHON" -m canopy2ground.train "$CONFIG" --seed "$SEED" --out "$OUT" \
    --data-root "$C2G_CACHE" --layer-root "$C2G_LAYERS" "$@"
