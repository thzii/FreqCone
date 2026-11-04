#!/usr/bin/env bash
# Mechanism-level sweep on SpecHier-48 (CPU, about 80 minutes).
#   Phase A  seed 0 (development): lambda_ent in {0, 0.03, 0.1, 0.3, 1}; pick lambda* by the
#            clustering accuracy on held-out KNOWN-class validation images (k-means with one
#            cluster per known class + Hungarian), ties broken toward the smaller lambda.
#            No novel-class label is used for selection. Rule fixed before seeing any result.
#   Phase B  seeds 1-3 (held out): gcd, hypcd, FreqCone(lambda*), FourEx-style partition, naive low-pass.
#   Phase C  seeds 1-3: FreqCone with lambda_ent = 1 (strong-hierarchy end of the trade-off).
set -e
cd "$(dirname "$0")"
mkdir -p results/results
cp results/pilot/*.json results/results/ 2>/dev/null || true
for w in 0 0.03 0.1 0.3 1; do
  python3 toy_experiment.py --method freqcone --seed 0 --w_ent "$w" --tag "_w$w" --out results/lambda_sweep
done
WSTAR=$(python3 - <<'EOF'
import glob, json
rows = [json.load(open(f))["metrics"] for f in glob.glob("results/lambda_sweep/*.json")]
best = sorted(rows, key=lambda m: (-round(m["val_cluster"], 6), m["w_ent"]))[0]
print(best["w_ent"])
EOF
)
echo "selected lambda_ent = $WSTAR" | tee results/lambda_sweep/selected.txt
for s in 1 2 3; do
  for m in gcd hypcd fourex_partition naive_lowpass; do
    python3 toy_experiment.py --method "$m" --seed "$s" --out results/results
  done
  python3 toy_experiment.py --method freqcone --seed "$s" --w_ent "$WSTAR" --tag "_sel" --out results/results
done
for s in 1 2 3; do
  python3 toy_experiment.py --method freqcone --seed "$s" --w_ent 1 --tag "_w1" --out results/results
done
echo DONE
