#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

OSM_FOLDER="data/oskar/skymodels/hogbom_experiments"
OUT_FOLDER="data/simulated/hogbom_experiments"
LOG_DIR="scripts/logs"
MAXJOBS="${MAXJOBS:-5}"

PY_SCRIPT="src/simulation/osm_to_fits.py"
NPIX="1024"
FOV_DEG="1"
RA0="-120"
DEC0="-60"

mkdir -p "$LOG_DIR"

models=( "$OSM_FOLDER"/*.osm )
(( ${#models[@]} ))|| { echo "no .osm in ${OSM_FOLDER}." >&2; exit 1;}

for osm in "${models[@]}"; do
    name="$(basename "${osm%.osm}")"

    while (( $(jobs -rp | wc -l) >= MAXJOBS )); do wait -n || true; done
    echo "Converting $name to fits."
    (
        if python $PY_SCRIPT --out "${OUT_FOLDER}/${name}" --osm  "${osm}" --npix $NPIX --fov-deg $FOV_DEG --dec0 "$DEC0" --ra0 "$RA0" > "${LOG_DIR}/${name}.log" 2>&1;then
            echo "DONE : $name"
        else
            echo "FAIL : $name"
        fi 
    ) &
done

wait
echo "Done."