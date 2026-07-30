#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:-/sdd/bgi/wangzilu/00.motorcortex_stduio/motorcortex_proj/45.DT-CCC}"
ENV_SH="${ENV_SH:-/sdd/bgi/wangzilu/miniconda3/etc/profile.d/conda.sh}"

MANIFEST="${1:-$WORKDIR/region_layer_rds/manifest.tsv}"
MAX_JOBS="${2:-1}"
RADIUS="${3:-100}"
DB="${4:-custom}"
SPECIES="${5:-human}"
FORMAT="${6:-raw}"
CUSTOM_DB="${7:-$WORKDIR/lr_db_human_niches_custom.csv}"
RUN_ONE_SCRIPT="${RUN_ONE_SCRIPT:-$WORKDIR/run_long_table/03_run_one_region_layer_niches_celltocellspatial.R}"

OUTDIR="$WORKDIR/niches_region_layer_radius${RADIUS}_${DB}_${SPECIES}_${FORMAT}_celltocellspatial"
LOGDIR="$OUTDIR/logs"

mkdir -p "$OUTDIR" "$LOGDIR"
source "$ENV_SH"
conda activate deeptalk-env

tail -n +2 "$MANIFEST" | while IFS=$'\t' read -r region_layer safe n_cells file; do
  out_prefix="$OUTDIR/niches.${safe}"
  log="$LOGDIR/${safe}.log"
  if [[ -s "${out_prefix}.rds" && -s "${out_prefix}.summary.txt" ]]; then
    echo "SKIP existing $region_layer cells=$n_cells"
    continue
  fi
  echo "START $region_layer cells=$n_cells"
  (
    cd "$WORKDIR"
    Rscript "$RUN_ONE_SCRIPT" \
      "$file" "$out_prefix" "$RADIUS" "$DB" "$SPECIES" "$FORMAT" "$CUSTOM_DB"
  ) >"$log" 2>&1 &

  while [[ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$MAX_JOBS" ]]; do
    wait -n
  done
done

wait
echo "DONE all region_layer NICHES CellToCellSpatial jobs"
