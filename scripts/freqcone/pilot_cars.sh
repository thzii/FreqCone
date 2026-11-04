#!/usr/bin/env bash
# Stanford Cars plan (docs/PLAN.md, section 3). Written before any SSB run; every rule below is fixed in advance.
#
# BASE METHOD  Hyp-SelEx from the official HypCD code (marked FreqCone blocks), label-order bug fixed
#              (--fix_label_order), DINOv2 ViT-B/14 with registers (dinov2_vitb14_reg4_pretrain.pth; its SHA-256 is
#              recorded in every run.json). Upstream DINOv2 configuration for Stanford Cars for both arms
#              (scripts/freqcone/hypselex_args.sh): batch 128, last 2 blocks, 200 epochs, lr 0.1, weight decay 5e-5,
#              D = 8192, c 0.1, r 2.0, smoothing 0.5, hyper_max_weight 0.5; O(N^2) distance (same values as upstream).
# FREQCONE     v2, lambda_dir 0.5, lambda_con 0.05, tau_a 0.5, tau_b 0.3, K 0.1; cut-offs from cutoffs.json (scars);
#              coarse resolution from runs/profile_scars/decision.json.
# EVALUATION   validation and test after every epoch (isolated random state); test numbers only in
#              runs/<id>/epochs_test.jsonl. Gates and conclusions use the LAST EPOCH only.
#
# LEVEL A (decision.json level A: one run <= 10 GPU-hours)
#   pilot, seed 7:  R0 baseline;  R1 FreqCone lambda_ent 0.03, A1;  R2 lambda 0.1, A1;  R3 lambda 0.1, A3
#   confirmation:   base_s8, base_s9, base_s10;  fc_s8, fc_s9, fc_s10 with the selected configuration
# LEVEL B (otherwise): no pilot; FreqCone fixed at lambda_ent 0.03, A1, v2 ("not tuned on SSB");
#   base_s8, base_s9, base_s10;  fc_s8, fc_s9, fc_s10.  No G1; G0 on base_s8.
# CONDITIONAL      R0-repro: upstream as is (label-order bug kept, no validation split), upstream's default seed 1,
#                  only if G0 fails.
#
# G0      The "upstream rule" row (epoch with the highest Old accuracy on the disjoint test split) of R0 (level A) or
#         base_s8 (level B): |All - 83.8| <= 2.0 (Hyp-SelEx, Stanford Cars, DINOv2: 83.8 / 93.3 / 79.2, HypCD
#         Table 1). Fail -> run R0-repro: if it passes, record that the gap comes from the bug fix and go on; if it
#         also fails, stop and report.
# SELECT  (level A) freqcone/select_pilot.py on R1-R3: validation metric at the last epoch, 1-SE rule with a paired,
#         class-stratified bootstrap (B = 2000, seed 0); ties go to the smaller lambda_ent, then the wider cone.
#         selection.json is committed before any test number of R1-R3 is read.
# G1      (level A) selected run vs R0, last epoch: dNew >= +1.0 and dAll >= -0.5. Fail -> stop the accuracy branch
#         and report (branch B of the earlier plan is then the option to discuss).
# FINAL   seeds 8-10, last epoch: FreqCone contributes on Stanford Cars if mean dNew >= +1.0, it wins on New for at
#         least 2 of 3 seeds, and mean dAll >= -0.5. A negative result is reported in full.
#
# TROUBLESHOOTING (no configuration change on our own): a non-finite loss stops the run (report it); out of memory
# -> resolution rule, else report; an interrupted run resumes from runs/<id>/ckpt_resume.pt (an incomplete run
# without a checkpoint is moved aside and rerun with the same seed); sharp views shallower than delta(b) is a
# result, not an error; more than 30 % over the planned time -> stop and report.
#
# usage (on a pilot-v2 tag, after compute_cutoffs, the smoke test and the profile): bash scripts/freqcone/pilot_cars.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
DATASET=scars
RUNS=runs
DECISION=$RUNS/profile_scars/decision.json
source scripts/freqcone/hypselex_args.sh

python -m freqcone.pilot_tools preflight "$RUNS" "$DECISION" --smoke_dir "$RUNS/smoke_scars"
if [ ${#FC_CUTS[@]} -eq 0 ]; then echo "cutoffs.json has no scars entry" >&2; exit 1; fi
read -r LEVEL COARSE < <(python -c "import json; d = json.load(open('$DECISION')); print(d['level'], d['coarse_res_flag'])")
echo "level $LEVEL, coarse views at ${COARSE:-224} px (0 = 224)"
DIAG=("${FC_CUTS[@]}" --coarse_res "$COARSE" --fc_K 0.1)         # for baselines: geometry diagnostics only
FC=(--freqcone True --freqcone_variant v2 "${DIAG[@]}" --fc_w_dir 0.5 --fc_w_con 0.05 --fc_tau_a 0.5 --fc_tau_b 0.3)

run() {
  local id=$1 seed=$2; shift 2
  local dir="$RUNS/$id" resume=False attempt=1
  if [ -f "$dir/run.json" ]; then echo "== $id: done, skipped"; return; fi
  if [ -f "$dir/ckpt_resume.pt" ]; then
    resume=True
  elif [ -d "$dir" ]; then
    mv "$dir" "$dir.incomplete_$(date +%Y%m%d_%H%M%S)"      # kept, not deleted; rerun from scratch, same seed
  fi
  if [ -f "$RUNS/.attempts_$id" ]; then attempt=$(( $(cat "$RUNS/.attempts_$id") + 1 )); fi
  echo "$attempt" > "$RUNS/.attempts_$id"
  mkdir -p "$dir"
  echo "== $id (seed $seed, attempt $attempt, resume $resume, $(date))"
  python -m train.train_HypSelEx "$@" --seed "$seed" --run_id "$id" --run_tag "scars_$id" --run_attempt "$attempt" \
      --run_dir "$dir" --resume "$resume" >> "$dir/train.log" 2>&1
  python -m freqcone.pilot_tools check "$dir"
  python -m freqcone.pilot_tools budget "$RUNS" "$DECISION" || { echo "over budget: report before going on"; exit 3; }
}

g0_or_repro() {
  if python -m freqcone.pilot_tools g0 "$1"; then return 0; fi
  echo "G0 failed on $1: running R0-repro (upstream as is, seed 1)"
  run R0-repro 1 "${UPSTREAM[@]}" "${DIAG[@]}" --eval_every_epoch True --hide_test_stdout True
  if python -m freqcone.pilot_tools g0 "$RUNS/R0-repro"; then
    echo "R0-repro passes G0: the gap of $1 comes from the label-order fix; going on" | tee "$RUNS/g0_note.txt"
    return 0
  fi
  echo "R0-repro fails G0 too: stop and report"
  exit 2
}

if [ "$LEVEL" = A ]; then
  run R0 7 "${UPSTREAM[@]}" "${PILOT[@]}" "${DIAG[@]}"
  g0_or_repro "$RUNS/R0"
  run R1 7 "${UPSTREAM[@]}" "${PILOT[@]}" "${FC[@]}" --w_ent 0.03 --aperture A1
  run R2 7 "${UPSTREAM[@]}" "${PILOT[@]}" "${FC[@]}" --w_ent 0.1 --aperture A1
  run R3 7 "${UPSTREAM[@]}" "${PILOT[@]}" "${FC[@]}" --w_ent 0.1 --aperture A3
  if [ ! -f "$RUNS/selection.json" ]; then
    python -m freqcone.select_pilot "$RUNS" > "$RUNS/selected.txt"
    git add "$RUNS/selection.json"
    git commit -m "chore: lock the Stanford Cars pilot selection (validation only)" -- "$RUNS/selection.json"
  fi
  read -r SEL W A < <(python -c "import json; s = json.load(open('$RUNS/selection.json'))['selected']; print(s['run_id'], s['w_ent'], s['aperture'])")
  python -m freqcone.compare_runs --base "$RUNS/R0" --fc "$RUNS/$SEL" --gate g1 --out "$RUNS/g1.json" \
    || { echo "G1 failed: stop the accuracy branch and report"; exit 4; }
  run base_s8 8 "${UPSTREAM[@]}" "${PILOT[@]}" "${DIAG[@]}"
else
  W=0.03; A=A1                                               # level B: fixed, not tuned on SSB
  run base_s8 8 "${UPSTREAM[@]}" "${PILOT[@]}" "${DIAG[@]}"
  g0_or_repro "$RUNS/base_s8"
fi
run base_s9 9 "${UPSTREAM[@]}" "${PILOT[@]}" "${DIAG[@]}"
run base_s10 10 "${UPSTREAM[@]}" "${PILOT[@]}" "${DIAG[@]}"
for s in 8 9 10; do
  run "fc_s$s" "$s" "${UPSTREAM[@]}" "${PILOT[@]}" "${FC[@]}" --w_ent "$W" --aperture "$A"
done
python -m freqcone.compare_runs --base "$RUNS"/base_s8 "$RUNS"/base_s9 "$RUNS"/base_s10 \
    --fc "$RUNS"/fc_s8 "$RUNS"/fc_s9 "$RUNS"/fc_s10 --gate final --plot "$RUNS/compare_final.png" \
    --out "$RUNS/compare_final.json" | tee "$RUNS/compare_final.md"
