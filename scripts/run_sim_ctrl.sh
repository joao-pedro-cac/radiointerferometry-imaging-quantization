#!/usr/bin/env bash
# Simulate the control scenario (ε=0, no outliers).
#
# Telescope:  meerkat.tm  (full 64-antenna MeerKAT, 2016 baselines)
# Observation: 1 h, 360 time steps (10 s cadence), 4 channels at 1.4 GHz
# Thermal floor: σ₀/√(725760 × 4 × 2 pol) ≈ 4.15e-8 Jy/beam
#
# Output: data/simulated/point_extended/

set -euo pipefail

# CONDA_BASE="$HOME/miniforge3"
# ENV="pfb"

PYTHON="python"
# Native binary (default) or Singularity/Apptainer image (.sif/.simg).
# Override at launch:  OSKAR=/path/to/oskar.sif bash run_sim_ctrl.sh

REPO="/home/mhiriy/radiointerferometry-imaging-quantization"
OUTPUT_DIR="$REPO/data/simulated"
SIMULATE="$REPO/src/simulation/simulate.py"
PFB_INIT="$REPO/src/simulation/pfb_init_noray.py"
NUMBA_THREADING_LAYER="omp"

OSKAR="$REPO/lib/OSKAR-2.12.2-Python3.sif"

TELESCOPE="meerkat.tm"
SKYMODEL=$1
RA=-120.0
DEC=-60.0
START_FREQ=1.4e9
NCHAN=4
FREQ_INC=1e6
OBS_LENGTH=3600
NTIME=360
SIGMA0=1e-4
FOV_DEG=1.0
SR_FACTOR=2.0
NTHREADS=8

VENV_PATH="/home/mhiriy/robust-radio-imaging/.venv"
source "${VENV_PATH}/bin/activate"

echo "[$(date +%H:%M:%S)] Simulating $SKYMODEL (ε=0) ..."
python "$SIMULATE" \
    --scenario-name    $SKYMODEL \
    --telescope        "$TELESCOPE" \
    --skymodel         "$SKYMODEL.osm" \
    --ra-deg           "$RA" \
    --dec-deg          "$DEC" \
    --start-freq-hz    "$START_FREQ" \
    --nchan            "$NCHAN" \
    --freq-inc-hz      "$FREQ_INC" \
    --obs-length-s     "$OBS_LENGTH" \
    --ntime-steps      "$NTIME" \
    --sigma0           "$SIGMA0" \
    --outlier-fraction 0.0 \
    --outlier-scale    1.0 \
    --seed             42 \
    --oskar            "$OSKAR" \
    --fov-deg          "$FOV_DEG" \
    --sr-factor        "$SR_FACTOR" \
    --output-dir       "$OUTPUT_DIR" \
    --data-root        "$REPO" \
    --nthreads         "$NTHREADS" \
    --overwrite

echo "[$(date +%H:%M:%S)] pfb init for $SKYMODEL..."
"$PYTHON" "$PFB_INIT" \
    --ms             "$OUTPUT_DIR/$SKYMODEL/obs.ms" \
    --output-prefix  "$OUTPUT_DIR/$SKYMODEL/obs" \
    --nthreads       "$NTHREADS"

echo "[$(date +%H:%M:%S)] Done. Data at: $OUTPUT_DIR/$SKYMODEL"
