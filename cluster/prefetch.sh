#!/bin/bash
# Phase 1 of the map, on a login node (the only nodes with internet): fetch every tile's imagery into a store,
# then optionally submit the offline GPU phase.
#     setsid nohup bash cluster/prefetch.sh maps/store_east_maui [maps/east_maui_dtm.tif] \
#         > logs/prefetch.log 2>&1 < /dev/null &
# Work is done in short passes of at most MAX_ROWS tile rows, so no process lives long enough for a login node's
# resource limits to kill it; each pass resumes where the last stopped. Completion is decided by counting
# row files against grid.json, never by reading a log. Five passes in a row without progress abort the run.
set -o pipefail
cd "$(dirname "$0")/.."
source cluster/env.sh
STORE=${1:?store dir}; OUT=${2:-}
PRESET=${PRESET:-east_maui}; MAX_ROWS=${MAX_ROWS:-8}; WORKERS=${WORKERS:-4}
log() { echo "[$(date '+%F %T')] $*"; }
rows() { find "$STORE" -maxdepth 1 -name 'row_*.npz' 2>/dev/null | wc -l; }
total() {    # rows in the grid; -1 until the first pass has written grid.json
  "$C2G_PYTHON" -c "import json, sys; print(json.load(open(sys.argv[1]))['ny'])" "$STORE/grid.json" 2>/dev/null \
    || echo -1
}
stall=0
while true; do
  before=$(rows); need=$(total)
  [ "$need" -ge 0 ] && [ "$before" -ge "$need" ] && break
  log "pass: $before/$need rows present"
  "$C2G_PYTHON" -u -m canopy2ground.prefetch --preset "$PRESET" --prior "$C2G_PRIOR" --out "$STORE" --layers vivid naip \
      --max-rows "$MAX_ROWS" --workers "$WORKERS"
  rc=$?; after=$(rows)
  log "pass ended with exit code $rc: $after/$(total) rows"
  if [ "$after" -le "$before" ]; then
    stall=$((stall + 1))
    [ "$stall" -ge 5 ] && { log "no progress in 5 passes; not submitting the GPU phase"; exit 1; }
    sleep 60
  else
    stall=0
  fi
done
log "store complete: $(rows) rows ($(du -sh "$STORE" | cut -f1))"
if [ -n "$OUT" ]; then
  log "GPU phase submitted: job $(sbatch --parsable cluster/predict.slurm "$STORE" "$OUT")"
fi
