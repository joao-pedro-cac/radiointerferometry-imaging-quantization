#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

OSM_FOLDER="/home/mhiriy/radiointerferometry-imaging-quantization/data/oskar/skymodels/hogbom_experiments"
NUMBA_THREADING_LAYER="omp"
EXP_ROOT="hogbom_experiments"
LOG_DIR="data/simulated/${EXP_ROOT}/logs"
MAXJOBS="${MAXJOBS:-5}"

mkdir -p "$LOG_DIR"

models=( "$OSM_FOLDER"/*.osm )          # .osm only — the folder also holds *.truth.csv
(( ${#models[@]} )) || { echo "no .osm in $OSM_FOLDER" >&2; exit 1; }
echo "launching ${#models[@]} sims, ${MAXJOBS} at a time"

for osm in "${models[@]}"; do
    name="$(basename "${osm%.osm}")"
    # throttle: block until fewer than MAXJOBS children are running
    while (( $(jobs -rp | wc -l) >= MAXJOBS )); do wait -n || true; done
    echo "start  $name"
    (
        if scripts/run_sim_ctrl.sh "${EXP_ROOT}/${name}" >"${LOG_DIR}/${name}.log" 2>&1; then
            echo "done   $name"
        else
            echo "FAIL   $name (see ${LOG_DIR}/${name}.log)" >&2
        fi
    ) &
done

wait
echo "all sims complete"