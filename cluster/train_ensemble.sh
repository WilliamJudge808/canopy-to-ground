#!/bin/bash
# Submit the six published ensemble members as two dependency lanes, so at most two run at once.
#     bash cluster/train_ensemble.sh [first-job-dependency]
# Each lane's jobs start when the previous one ends (afterany), so one failure does not stall the rest.
set -eo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs runs
dep=${1:-}
sub() {    # sub <after-job-id or ""> <config> <seed>
  local after=$1 cfg=$2 seed=$3 name
  name="$(basename "$cfg" .yaml)_s$seed"
  sbatch --parsable -J "$name" ${after:+--dependency=afterany:$after} \
      cluster/train.slurm "$cfg" "$seed" "runs/$name"
}
a=$(sub "$dep" configs/b5_coast.yaml 0); a=$(sub "$a" configs/b5.yaml 0); a=$(sub "$a" configs/b2.yaml 0)
b=$(sub "$dep" configs/b5_coast.yaml 1); b=$(sub "$b" configs/b5.yaml 1); b=$(sub "$b" configs/b2.yaml 1)
echo "lane 1 ends with job $a, lane 2 with job $b"
