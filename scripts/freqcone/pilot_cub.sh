#!/usr/bin/env bash
# P1: SSB pilot of FreqCone on CUB, one development seed. Protocol written on 2026-11-04, BEFORE any SSB run.
#
# BASE METHOD  Hyp-SelEx from the official HypCD code (commit ffc895b + the marked FreqCone blocks), DINOv2
#              ViT-B/14 with registers. Every run uses the upstream DINOv2 configuration for CUB
#              (scripts/freqcone/hypselex_args.sh): batch 128, last 2 blocks, 200 epochs, lr 0.1, D = 8192,
#              c = 0.1, r = 1.5, unsupervised smoothing 1.0, hyper_max_weight 1.0; O(N^2) distance (same values
#              as upstream). Seed 7. Seeds 8, 9, 10 are reserved for P2 and seeds 11-13 for branch B.
# CUT-OFFS     f_a, f_b from cutoffs.json (SNR 5 / 10 dB, FourEx's definition, training images only, no labels).
# RESOLUTION   coarse views at 224 or 112 px per decision.json (rule in freqcone/pilot_tools.py, fixed before R1).
#
# RUNS (sequential, same tagged commit; K = 0.1, lambda_dir = 0.5, lambda_con = 0.05, tau_a = 0.5, tau_b = 0.3)
#   R0-repro  upstream exactly: SelEx label-order bug kept, 100 % of the labels, no validation split. G0 only.
#   R0        baseline with the pilot flags: validation split held out, label-order bug fixed (SelEx issue #11).
#   R1        FreqCone v2, lambda_ent = 0 (L_dir + L_con only; the aperture has no effect, run once with A1)
#   R2 R3 R4  FreqCone v2, lambda_ent = 0.03 with apertures A1, A2, A3
#   R5 R6 R7  FreqCone v2, lambda_ent = 0.1  with apertures A1, A2, A3
#   R8        FreqCone v3 at the lambda_ent > 0 configuration chosen by the rule below (mechanism analysis only)
#   Apertures (r_a, r_b): A1 (0.120, 0.200), A2 (0.104, 0.136), A3 (0.100, 0.114). With c = 0.1 their coarse depths
#   delta(b) are 1.282, 0.866 and 0.724. If decision.json drops A2 (budget above 100 GPU-hours), R3 and R6 are not run.
#
# VALIDATION METRIC  Hyp-SelEx predicts with semi-supervised k-means (upstream test()); it has no classifier head.
#   The metric applies that procedure to the held-out validation images (20 % of the labelled images of each
#   known class, never trained on, identical across runs: val_list_sha256): upstream K_Means with the settings of
#   test(), the labelled training images as fixed anchors and the validation images as unlabelled points,
#   k = number of known classes, L2-normalised backbone features (test transform). Scored with upstream
#   cluster_acc, the Hungarian accuracy that split_cluster_acc_v2 reports as "All". Per-image predictions are
#   stored. No novel-class label and no test-split number is used.
#
# SELECTION RULE (freqcone/select_pilot.py, unit-tested in tests_freqcone/test_select_pilot.py)
#   Candidates: the FreqCone runs R1-R7. R0 is the reference, not a candidate.
#   Score: s_m = the validation metric of run m at the last epoch.
#   Bootstrap: B = 2000, bootstrap seed 0, resampling with replacement within each class (stratified), the same
#   resampled index sets for every candidate (paired); every replicate recomputes the whole metric, Hungarian
#   matching included.
#   m* = argmax s_m. For each m: d_m(b) = s_{m*}(b) - s_m(b) and SE_m = std_b(d_m(b)). m is tied with m* if
#   s_{m*} - s_m <= SE_m. Ties go to the simpler run: smaller lambda_ent first (0 < 0.03 < 0.1), then the
#   wider cone (A3 < A2 < A1).
#   Output: selection.json with every s_m, SE_m, the tie set, the choice, and the SHA-256 of every input file.
#   It is committed before any script prints a test number of R1-R7 (report_pilot.py refuses otherwise).
#   R8 configuration: the same rule restricted to the candidates with lambda_ent > 0 (R2-R7). If the rule
#   selects R1, that is reported as is: the selected FreqCone is L_dir + L_con only.
#
# GATES
#   G0 (after R0-repro)  |upstream_best All - 90.7| <= 2.0 (Hyp-SelEx, CUB, DINOv2, HypCD Table 1). Fail: stop,
#                        check the SSB split, c / r, augmentation, epochs and batch, and report before changing
#                        anything. upstream_best = upstream's checkpoint chosen on test Old accuracy, used here only.
#   G1 (after locking)   selected vs R0, last-epoch test numbers (v2): pass if New_sel - New_R0 >= +1.0 and
#                        All_sel - All_R0 >= -0.5. One seed: a cheap filter, not evidence. Pass -> P2; fail ->
#                        stop the accuracy branch and report.
#
# QUESTIONS FIXED IN ADVANCE
#   Q1 regime    Is R0's mean sharp-view depth above delta(b) of A1, A2, A3 (deep regime: the cone can act)?
#                Prediction from DINOHead's unit-norm rows: ||v||^2 ~ D / 256 = 32 > r^2, so most samples are
#                clipped at depth 2r = 3.0 > 1.282. Untested.
#   Q2 accuracy  Paired All / Old / New of the selected configuration against R0 (G1).
#   Q3 cost      Seconds per epoch and peak memory of FreqCone against R0.
#
# TROUBLESHOOTING (no configuration change on our own): a non-finite loss stops the run (report it); out of
#   memory -> rule of step 3.3, else report; an interrupted run restarts from scratch with the same seed and its
#   attempt number is recorded; sharp views shallower than delta(b) is a result, not an error; more than 30 %
#   over the planned time -> stop and report.
#
# usage (on a pilot-v* tag, after compute_cutoffs, smoke test and profile): bash scripts/freqcone/pilot_cub.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
DATASET=cub
SEED=7
OUT=results_ssb/pilot_cub
DECISION=results_ssb/profile_cub/decision.json
mkdir -p "$OUT/logs"
source scripts/freqcone/hypselex_args.sh

python -m freqcone.pilot_tools preflight "$OUT" "$DECISION" --smoke_dir results_ssb/smoke_cub
if [ ${#FC_CUTS[@]} -eq 0 ]; then echo "cutoffs.json has no cub entry" >&2; exit 1; fi
read -r COARSE DROP_A2 < <(python -c "import json; d = json.load(open('$DECISION')); print(d['coarse_size_flag'], int(d['drop_A2']))")
# DIAG: cut-offs and coarse resolution. For the baselines they only set the geometry diagnostics (same coarse
# views as the FreqCone runs); they do not change training.
DIAG=("${FC_CUTS[@]}" --fc_coarse_size "$COARSE" --fc_K 0.1)
FC=(--freqcone True "${DIAG[@]}" --fc_w_dir 0.5 --fc_w_con 0.05 --fc_tau_a 0.5 --fc_tau_b 0.3)

run() {
  local id=$1; shift
  local json="$OUT/$id.json"
  if [ -f "$json" ]; then echo "== $id: done, skipped"; return; fi
  local attempt=1
  if [ -f "$OUT/$id.attempts" ]; then attempt=$(( $(cat "$OUT/$id.attempts") + 1 )); fi
  echo "$attempt" > "$OUT/$id.attempts"
  echo "== $id (attempt $attempt, $(date)); log: $OUT/logs/$id.log (do not open R1-R8 logs before the selection is committed)"
  python -m train.train_HypSelEx "$@" --run_id "$id" --run_tag "pilot_cub_$id" --run_attempt "$attempt" \
      --results_json "$json" > "$OUT/logs/$id.log" 2>&1
  python -m freqcone.pilot_tools check "$json"
  python -m freqcone.pilot_tools budget "$OUT" "$DECISION" || { echo "over budget: report before going on"; exit 3; }
}

run R0-repro "${UPSTREAM[@]}" "${DIAG[@]}"
python -m freqcone.pilot_tools g0 "$OUT/R0-repro.json" || { echo "G0 failed: stop and report"; exit 2; }

run R0 "${UPSTREAM[@]}" "${PILOT[@]}" "${DIAG[@]}"
run R1 "${UPSTREAM[@]}" "${PILOT[@]}" "${FC[@]}" --fc_w_ent 0 --fc_aperture A1
run R2 "${UPSTREAM[@]}" "${PILOT[@]}" "${FC[@]}" --fc_w_ent 0.03 --fc_aperture A1
[ "$DROP_A2" = 1 ] || run R3 "${UPSTREAM[@]}" "${PILOT[@]}" "${FC[@]}" --fc_w_ent 0.03 --fc_aperture A2
run R4 "${UPSTREAM[@]}" "${PILOT[@]}" "${FC[@]}" --fc_w_ent 0.03 --fc_aperture A3
run R5 "${UPSTREAM[@]}" "${PILOT[@]}" "${FC[@]}" --fc_w_ent 0.1 --fc_aperture A1
[ "$DROP_A2" = 1 ] || run R6 "${UPSTREAM[@]}" "${PILOT[@]}" "${FC[@]}" --fc_w_ent 0.1 --fc_aperture A2
run R7 "${UPSTREAM[@]}" "${PILOT[@]}" "${FC[@]}" --fc_w_ent 0.1 --fc_aperture A3

if [ ! -f "$OUT/selection.json" ]; then
  python -m freqcone.select_pilot "$OUT" > "$OUT/r8_config.txt"
  git add "$OUT/selection.json"
  git commit -m "chore: lock CUB pilot selection (validation only)" -- "$OUT/selection.json"
fi
read -r W8 A8 < <(python -c "import json; r = json.load(open('$OUT/selection.json'))['r8_config']; print(r['w_ent'], r['aperture'])")
run R8 "${UPSTREAM[@]}" "${PILOT[@]}" "${FC[@]}" --fc_v3 True --fc_w_ent "$W8" --fc_aperture "$A8"

python -m freqcone.report_pilot "$OUT"
