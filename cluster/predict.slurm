#!/bin/bash
#SBATCH --job-name=c2g-predict
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x-%j.out
# Offline phase of the map: the ensemble over a complete imagery store, then the coastline mask.
#     sbatch cluster/predict.slurm <store_dir> <out.tif>      (from the repo root; logs/ must exist)
# The store comes from cluster/prefetch.sh (compute nodes have no internet). Members default to the six
# runs of cluster/train_ensemble.sh; set CHECKPOINTS to override. Writes <out.tif> and <out>_coast.tif.
set -eo pipefail
STORE=${1:?store dir}; OUT=${2:?output tif}
cd "${SLURM_SUBMIT_DIR:-.}"          # sbatch runs a spooled copy of this script
source cluster/env.sh
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
mkdir -p logs "$(dirname "$OUT")"
CHECKPOINTS=${CHECKPOINTS:-$(printf 'runs/%s/model.pt ' b2_s0 b2_s1 b5_s0 b5_s1 b5_coast_s0 b5_coast_s1)}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# shellcheck disable=SC2086  # CHECKPOINTS is a space-separated list
"$C2G_PYTHON" -m canopy2ground.predict --checkpoint $CHECKPOINTS --store "$STORE" --prior "$C2G_PRIOR" \
    --coastline "$C2G_COAST/mauico_detailedcoast_county.geojson" --out "$OUT" --shift-tta 4 --fused-attention
"$C2G_PYTHON" -m canopy2ground.mapping.coast_mask --pred "$OUT" --out "${OUT%.tif}_coast.tif" \
    --county "$C2G_COAST/mauico_detailedcoast_eastmaui.geojson" \
    --usace "$C2G_COAST/usace2013_lidar_landpoly_eastmaui.geojson" --prior "$C2G_PRIOR"
