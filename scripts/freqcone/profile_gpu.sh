#!/usr/bin/env bash
# Step 3.3, memory and speed probes at the upstream batch size (128) with the dataset's upstream c / r and the
# pilot flags. Each probe trains STEPS steps of one epoch, then runs upstream's per-epoch evaluation and
# checkpointing, and records peak memory and an estimated full-epoch time (results_ssb/profile_<dataset>/).
#   upstream  baseline with upstream dist_matrix (N x N x D)   lowmem  baseline, O(N^2) distance
#   fc224     FreqCone v2, coarse views at 224 px               fc112   FreqCone v2, coarse views at 112 px
# An out-of-memory failure is a result: it is logged and the next probe runs.
# Then `pilot_tools decide` applies the fixed rule (resolution, budget, A2) and writes decision.json and the
# README section. Commit both before the pilot.
# usage: DATASET=cub STEPS=30 bash scripts/freqcone/profile_gpu.sh
set -u
cd "$(dirname "$0")/../.."
DATASET=${DATASET:-cub}
SEED=${SEED:-7}
STEPS=${STEPS:-30}
OUT=results_ssb/profile_${DATASET}
mkdir -p "$OUT"
source scripts/freqcone/hypselex_args.sh
if [ ${#FC_CUTS[@]} -eq 0 ]; then echo "cutoffs.json has no $DATASET entry: run scripts/freqcone/compute_cutoffs.py first" >&2; exit 1; fi

run() {
  local tag=$1; shift
  echo "== $tag"
  python -m train.train_HypSelEx "${UPSTREAM[@]}" "${PILOT[@]}" --epochs 1 --profile_steps "$STEPS" \
      --run_id "profile" --run_tag "$tag" --results_json "$OUT/$tag.json" "$@" > "$OUT/$tag.log" 2>&1 \
    || echo "   $tag FAILED (exit $?), see $OUT/$tag.log (look for 'out of memory')"
}

run upstream --dist_impl upstream "${FC_CUTS[@]}"
run lowmem "${FC_CUTS[@]}"
run fc224 --freqcone True "${FC_CUTS[@]}" --fc_w_ent 0.1 --fc_aperture A1
run fc112 --freqcone True "${FC_CUTS[@]}" --fc_w_ent 0.1 --fc_aperture A1 --fc_coarse_size 112
python -m freqcone.pilot_tools check --profile "$OUT"/*.json || true
python -m freqcone.pilot_tools decide "$OUT"
