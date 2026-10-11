# Shared paths for running canopy2ground on a Slurm cluster. Source it from the repo root:
#     source cluster/env.sh
# Home quotas on Princeton-style clusters are small, so everything large lives on scratch. Any variable can be
# set before sourcing to override its default.
export C2G_SCRATCH="${C2G_SCRATCH:-/scratch/network/$USER}"
export C2G_REPO="${C2G_REPO:-$C2G_SCRATCH/canopy-to-ground}"
export C2G_DATA="${C2G_DATA:-$C2G_REPO/data}"
export C2G_ENV="${C2G_ENV:-$C2G_SCRATCH/envs/canopy2ground}"      # conda env with the package installed
export C2G_PYTHON="${C2G_PYTHON:-$C2G_ENV/bin/python}"
export C2G_CONDA_MODULE="${C2G_CONDA_MODULE:-anaconda3/2025.12}"   # `module load` this to build the env

# Training data (see configs/*.yaml) and map inputs.
export C2G_CACHE="${C2G_CACHE:-$C2G_DATA/cache}"
export C2G_LAYERS="${C2G_LAYERS:-$C2G_DATA/layers}"
export C2G_PRIOR="${C2G_PRIOR:-$C2G_DATA/priors/copernicus}"      # Copernicus GLO-30 tiles
export C2G_COAST="${C2G_COAST:-$C2G_DATA/coast}"                  # coastline GeoJSONs, see scripts/reproduce_map.sh

# Caches that would otherwise fill $HOME. Compute nodes are offline, so model weights must be cached
# from a login node first (any `python -m canopy2ground.train` run there downloads them).
export HF_HOME="${HF_HOME:-$C2G_SCRATCH/hf_cache}"
export TORCH_HOME="${TORCH_HOME:-$C2G_SCRATCH/torch_cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$C2G_SCRATCH/pip_cache}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$C2G_SCRATCH/conda_pkgs}"
export PYTHONPATH="$C2G_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
