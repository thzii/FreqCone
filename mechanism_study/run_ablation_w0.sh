#!/usr/bin/env bash
# Ablation on the held-out seeds: FreqCone without the entailment term (lambda_ent = 0),
# i.e. HypCD + L_dir + L_con. Isolates what the cone adds on top of the other two losses.
set -e
cd "$(dirname "$0")"
for s in 1 2 3; do
  python3 toy_experiment.py --method freqcone --seed "$s" --w_ent 0 --tag _w0 --out results/results
done
echo ABLATION_DONE
