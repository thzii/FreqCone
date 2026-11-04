"""Aggregate toy_experiment.py results: mean +- std over seeds and paired deltas."""
import glob
import json
import sys
from collections import defaultdict

import numpy as np

ORDER = ["gcd", "hypcd", "fourex_partition", "naive_lowpass", "freqcone_v0", "freqcone_v1",
         "freqcone", "freqcone_w0", "freqcone_sel", "freqcone_w1"]
METRICS = ["all", "old", "new", "parent_acc", "novel_to_parent", "val_cluster"]


def load(dirs):
    res = defaultdict(dict)
    for d in dirs:
        for f in glob.glob(f"{d}/*.json"):
            m = json.load(open(f))["metrics"]
            res[m["method"]][m["seed"]] = m
    return res


def main(dirs, only_seeds=None):
    res = load(dirs)
    seeds = sorted(set().union(*[set(v) for v in res.values()]))
    if only_seeds:
        seeds = [s for s in seeds if s in only_seeds]
    print("seeds:", seeds)
    print(f"{'method':18s} " + " ".join(f"{k:>16s}" for k in METRICS) + "   n")
    rows = {}
    for meth in ORDER:
        if meth not in res:
            continue
        runs = [res[meth][s] for s in seeds if s in res[meth]]
        if not runs:
            continue
        vals = {k: np.array([r.get(k, np.nan) for r in runs]) * 100 for k in METRICS}
        rows[meth] = vals
        print(f"{meth:18s} " + " ".join(f"{v.mean():7.1f} ± {v.std(ddof=1) if len(v) > 1 else 0:5.1f}  " for v in vals.values()) + f"  {len(runs)}")
    print("\nPaired difference vs hypcd (same seed), New ACC and parent metrics:")
    base = res.get("hypcd", {})
    for meth in ORDER:
        if meth in ("hypcd",) or meth not in res:
            continue
        common = [s for s in seeds if s in res[meth] and s in base]
        if not common:
            continue
        for k in ("all", "new", "novel_to_parent"):
            d = np.array([res[meth][s][k] - base[s][k] for s in common]) * 100
            print(f"  {meth:18s} {k:16s} deltas {np.round(d, 1)}  mean {d.mean():+.1f}  wins {int((d > 0).sum())}/{len(d)}")
    print("\nGeometry diagnostics:")
    for meth in ORDER:
        for s in seeds:
            m = res.get(meth, {}).get(s)
            if m and "depth_full" in m:
                print(f"  {meth:14s} seed {s}: depth a {m['depth_a']:.2f}  b {m['depth_b']:.2f}  full {m['depth_full']:.2f}"
                      f"  ordered {m['frac_ordered']:.3f}  in_cone {m.get('in_cone_b_u', float('nan')):.3f}"
                      f"  angle(b,u) {m.get('angle_b_u_deg', float('nan')):.1f}")


if __name__ == "__main__":
    # usage: python aggregate.py results/results [--seeds 1,2,3]
    args = sys.argv[1:]
    only = None
    if "--seeds" in args:
        i = args.index("--seeds")
        only = [int(s) for s in args[i + 1].split(",")]
        args = args[:i] + args[i + 2:]
    main(args or ["results/results"], only)
