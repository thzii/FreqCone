#!/usr/bin/env bash
# Smoke test on the GPU machine (a few minutes; not a result). Two epochs on every 10th image of every split, with
# the per-epoch evaluation on, in both data modes of the plan:
#   with the validation split (pilot):
#     smoke_base  baseline                                 R1  FreqCone v2, 224 px, lambda_ent 0.03, A1
#     R3          FreqCone v2, 112 px, lambda 0.1, A3      smoke_v3  FreqCone v3, 112 px, lambda 0.1, A1
#   without it (confirmation):
#     B8          baseline                                 F8  FreqCone v2, 224 px, lambda_ent 0.03, A1
# Checks: every run finishes (a non-finite loss stops a run), run.json holds no test number, the JSONL files cover
# both epochs, the diagnostics are there, and select_pilot / summarize_run / compare_runs run on the records.
# Then delete the results: the pilot's preflight refuses to start while runs/smoke_<dataset> exists.
# usage: DATASET=scars bash scripts/freqcone/smoke_gpu.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
DATASET=${DATASET:-scars}
EPOCHS=2
OUT=runs/smoke_${DATASET}
mkdir -p "$OUT"
source scripts/freqcone/hypselex_args.sh
python -m freqcone.pilot_tools env
if [ ${#FC_CUTS[@]} -eq 0 ]; then echo "cutoffs.json has no $DATASET entry: run scripts/freqcone/compute_cutoffs.py first" >&2; exit 1; fi
SMOKE=(--smoke_frac 10 --fc_log_every 2 --seed 7)
FC=(--freqcone True "${FC_CUTS[@]}")

run() {
  local id=$1; shift
  echo "== $id"
  mkdir -p "$OUT/$id"
  python -m train.train_HypSelEx "${UPSTREAM[@]}" "${SMOKE[@]}" --run_id "$id" --run_tag "smoke_$id" \
      --run_dir "$OUT/$id" "$@" > "$OUT/$id/train.log" 2>&1 \
    || { echo "   $id FAILED, see $OUT/$id/train.log"; exit 1; }
}

run smoke_base "${PILOT[@]}" "${FC_CUTS[@]}"
run R1 "${PILOT[@]}" "${FC[@]}" --w_ent 0.03 --aperture A1
run R3 "${PILOT[@]}" "${FC[@]}" --w_ent 0.1 --aperture A3 --coarse_res 112
run smoke_v3 "${PILOT[@]}" "${FC[@]}" --freqcone_variant v3 --w_ent 0.1 --aperture A1 --coarse_res 112
run B8 "${CONFIRM[@]}" "${FC_CUTS[@]}"
run F8 "${CONFIRM[@]}" "${FC[@]}" --w_ent 0.03 --aperture A1
python -m freqcone.pilot_tools check "$OUT"/smoke_base "$OUT"/R1 "$OUT"/R3 "$OUT"/smoke_v3 "$OUT"/B8 "$OUT"/F8
python -m freqcone.select_pilot "$OUT" --n_boot 200 > /dev/null
python -m freqcone.summarize_run "$OUT"/smoke_base > /dev/null
python -m freqcone.summarize_run "$OUT"/B8 > /dev/null
python -m freqcone.compare_runs --base "$OUT"/B8 --fc "$OUT"/F8 > /dev/null
echo "smoke OK. Delete it before the pilot: rm -rf $OUT"
