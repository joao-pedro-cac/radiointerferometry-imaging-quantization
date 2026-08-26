#!/usr/bin/env bash
# Run the image processing pipeline with a simulation configuration JSON file
# and outputs a directory containing the computation metrics and results
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

CONFIG_DIR="configs/hogbom_experiment"     # e.g. configs/hogbom
PIPELINE="src/main.py"           # e.g. src/.../pipeline.py
LOG_DIR="logs/hogbom_experiment"
MAXJOBS="${MAXJOBS:-20}"

mkdir -p "$LOG_DIR"
configs=( "$CONFIG_DIR"/*.json )
(( ${#configs[@]} )) || { echo "no .json in $CONFIG_DIR" >&2; exit 1; }
echo "launching ${#configs[@]} configs, ${MAXJOBS} at a time"

N_DONE=0
N_FAIL=0
for cfg in "${configs[@]}"; do
    name="$(basename "${cfg%.json}")"
    while (( $(jobs -rp | wc -l) >= MAXJOBS )); do wait -n || true; done
    echo "start  $name"
    (
        if python -u "$PIPELINE" "$cfg" > "${LOG_DIR}/${name}.log" 2>&1; then
            echo "DONE  $name"
            N_DONE=$(($N_DONE + 1))
            echo "$N_DONE experiments completed."
        else
            echo "FAIL  $name (see ${LOG_DIR}/${name}.log)" >&2
            N_FAIL=$(($N_FAIL + 1))
            echo "$N_FAIL experiments completed."
        fi
    ) &
done

wait
echo "batch complete, $N_DONE experiments succeed; $N_FAIL experiments failed."