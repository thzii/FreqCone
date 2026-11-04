#!/usr/bin/env bash
# FreqCone v3 study on FRESH seeds 4-6 (never run before). Written before any v3 result was seen.
#   v3 = v2 without the stop-gradient on the sharp-view norm in L_ent (see FREQCONE_VARIANTS).
#   lambda_ent is NOT re-selected: v3 uses lambda* = 0.03 chosen for v2 on seed 0, plus lambda = 1.
# Questions fixed in advance:
#   Q1 (mechanism) on seeds where HypCD keeps sharp views shallower than the coarse depth 1.81,
#      does v3 put sharp views inside their cones (in_cone_b_u > 0) where v2 cannot?
#   Q2 (accuracy)  paired All / New / Novel->parent of v3(lambda*) against HypCD and against v2(lambda*).
set -e
cd "$(dirname "$0")"
for s in 4 5 6; do
  python3 toy_experiment.py --method hypcd --seed "$s" --out results/results_v3
  python3 toy_experiment.py --method freqcone --seed "$s" --w_ent 0.03 --tag _sel --out results/results_v3
  python3 toy_experiment.py --method freqcone_v3 --seed "$s" --w_ent 0.03 --tag _sel --out results/results_v3
  python3 toy_experiment.py --method freqcone_v3 --seed "$s" --w_ent 1 --tag _w1 --out results/results_v3
done
echo V3_DONE
