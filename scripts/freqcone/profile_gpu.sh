#!/usr/bin/env bash
# Memory and speed probes (plan 3.1) at the upstream batch size (128) with the dataset's upstream c / r and the pilot
# flags, per-epoch evaluation included. Each probe trains STEPS steps of one epoch, then runs upstream's per-epoch
# evaluation and checkpointing and the per-epoch evaluation, and records peak memory and an estimated full-epoch
# time (runs/profile_<dataset>/<probe>/run.json):
#   upstream  baseline with upstream dist_matrix (N x N x D)   lowmem  baseline, O(N^2) distance
#   fc224     FreqCone v2, coarse views at 224 px               fc112   FreqCone v2, coarse views at 112 px
# An out-of-memory failure is a result: it is logged and the next probe runs.
# Then `pilot_tools decide` applies the fixed rule (resolution, level A / B) and writes decision.json and the
# README section. Commit both, then tag pilot-v2.
# usage: DATASET=scars STEPS=30 bash scripts/freqcone/profile_gpu.sh
set -u
cd "$(dirname "$0")/../.."
DATASET=${DATASET:-scars}
STEPS=${STEPS:-30}
OUT=runs/profile_${DATASET}
mkdir -p "$OUT"
source scripts/freqcone/hypselex_args.sh
if [ ${#FC_CUTS[@]} -eq 0 ]; then echo "cutoffs.json has no $DATASET entry: run scripts/freqcone/compute_cutoffs.py first" >&2; exit 1; fi

run() {
  local tag=$1; shift
  echo "== $tag"
  mkdir -p "$OUT/$tag"
  python -m train.train_HypSelEx "${UPSTREAM[@]}" "${PILOT[@]}" --seed 7 --epochs 1 --profile_steps "$STEPS" \
      --run_id profile --run_tag "$tag" --run_dir "$OUT/$tag" "$@" > "$OUT/$tag/train.log" 2>&1 \
    || echo "   $tag FAILED (exit $?), see $OUT/$tag/train.log (look for 'out of memory')"
}

run upstream --dist_impl upstream "${FC_CUTS[@]}"
run lowmem "${FC_CUTS[@]}"
run fc224 --freqcone True "${FC_CUTS[@]}" --w_ent 0.1 --aperture A1
run fc112 --freqcone True "${FC_CUTS[@]}" --w_ent 0.1 --aperture A1 --coarse_res 112
python -m freqcone.pilot_tools check --profile "$OUT"/upstream "$OUT"/lowmem "$OUT"/fc224 "$OUT"/fc112 || true
python -m freqcone.pilot_tools decide "$OUT"
