#!/usr/bin/env bash
# Step 3.2, smoke test (a few minutes; not a result). Two epochs on every 10th image of every split, for:
#   R0  baseline with the pilot flags          R2  FreqCone v2, 224 px, lambda_ent 0.03, A1
#   R7  FreqCone v2, 112 px, lambda 0.1, A3    R8  FreqCone v3, 112 px, lambda 0.1, A1
# Checks: every run finishes (a non-finite loss stops a run), the records pass the schema check including the
# depth_full / in_cone / clip_frac / angle_b_u diagnostics, and select_pilot runs on them.
# Then delete the results: the pilot's preflight refuses to start while results_ssb/smoke_<dataset> exists.
# usage: DATASET=cub bash scripts/freqcone/smoke_gpu.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
DATASET=${DATASET:-cub}
SEED=${SEED:-7}
EPOCHS=2
OUT=results_ssb/smoke_${DATASET}
mkdir -p "$OUT"
source scripts/freqcone/hypselex_args.sh
if [ ${#FC_CUTS[@]} -eq 0 ]; then echo "cutoffs.json has no $DATASET entry: run scripts/freqcone/compute_cutoffs.py first" >&2; exit 1; fi
SMOKE=(--smoke_frac 10 --fc_log_every 2)

run() {
  local id=$1; shift
  echo "== $id"
  python -m train.train_HypSelEx "${UPSTREAM[@]}" "${PILOT[@]}" "${SMOKE[@]}" --run_id "$id" --run_tag "smoke_$id" \
      --results_json "$OUT/$id.json" "$@" > "$OUT/$id.log" 2>&1 \
    || { echo "   $id FAILED, see $OUT/$id.log"; exit 1; }
}

run R0 "${FC_CUTS[@]}"
run R2 --freqcone True "${FC_CUTS[@]}" --fc_w_ent 0.03 --fc_aperture A1
run R7 --freqcone True "${FC_CUTS[@]}" --fc_w_ent 0.1 --fc_aperture A3 --fc_coarse_size 112
run R8 --freqcone True --fc_v3 True "${FC_CUTS[@]}" --fc_w_ent 0.1 --fc_aperture A1 --fc_coarse_size 112
python -m freqcone.pilot_tools check "$OUT"/R*.json
python -m freqcone.select_pilot "$OUT" --n_boot 200 > /dev/null
echo "smoke OK. Delete it before the pilot: rm -rf $OUT"
