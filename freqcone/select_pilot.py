"""Pre-registered selection for the Stanford Cars pilot (level A). Rule fixed before any SSB run.

This module reads ONLY the validation fields of runs/<id>/run.json (schema, run_id, dataset, seed, freqcone
configuration, val); run.json holds no test number anyway. Test numbers of R1-R3 are printed by summarize_run.py
and compare_runs.py, which refuse until this module's output is committed.

Candidates  FreqCone runs R1 (lambda_ent 0.03, A1), R2 (0.1, A1), R3 (0.1, A3), seed 7. R0 is the reference.
Score       s_m = val.value of run m at the last epoch: the base method's own prediction procedure on the
            held-out known-class validation images, scored with the test's Hungarian accuracy (README).
Bootstrap   B = 2000, seed 0. Resampling with replacement within each class (stratified), the same resampled
            index sets for every candidate (paired). Each replicate recomputes the whole metric, Hungarian
            matching included, from the stored per-image predictions.
Choice      m* = argmax s_m. For every m: d_m(b) = s_{m*}(b) - s_m(b), SE_m = std_b d_m(b) (ddof 1). m is tied
            with m* when s_{m*} - s_m <= SE_m. The simplest tied run wins: smaller lambda_ent first, then the
            wider cone (A3 < A2 < A1).
Output      RUNS_ROOT/selection.json with every s_m and SE_m, the tie set, the choice, B, the seed, the
            validation-list SHA-256 and the SHA-256 of every input file. Commit it before any test number of
            R1-R3 is read.

usage: python -m freqcone.select_pilot runs [--out PATH]
The last stdout line is "<lambda_ent> <aperture>" of the selected configuration (read by pilot_cars.sh).
"""
import argparse
import glob
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

from freqcone.hypcd import SCHEMA, VAL_METRIC
from freqcone.runs import CANDIDATES

APERTURE_RANK = {'A3': 0, 'A2': 1, 'A1': 2}          # wider cone (weaker constraint) first
VAL_FIELDS = ('schema', 'run_id', 'dataset', 'seed', 'freqcone', 'val')
B_DEFAULT, SEED_DEFAULT = 2000, 0


def hungarian_acc(y_true, y_pred):
    """Upstream's Hungarian accuracy (the 'All' of split_cluster_acc_v2); see tests_freqcone/test_hungarian.py."""
    from project_utils.cluster_utils import cluster_acc
    return float(cluster_acc(np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)))


def load_val_view(path):
    """Whitelisted fields of a run record and the SHA-256 of the file bytes."""
    with open(path, 'rb') as f:
        raw = f.read()
    record = json.loads(raw)
    view = {k: record.get(k) for k in VAL_FIELDS} if isinstance(record, dict) else {}
    return view, hashlib.sha256(raw).hexdigest()


def load_candidates(runs_root):
    views, hashes = [], {}
    for path in sorted(glob.glob(os.path.join(runs_root, '*', 'run.json'))):
        view, digest = load_val_view(path)
        if view.get('schema') != SCHEMA or view.get('run_id') not in CANDIDATES:
            continue
        rel = os.path.relpath(path, runs_root).replace('\\', '/')
        fc = view.get('freqcone') or {}
        if not fc.get('enable') or fc.get('variant') != 'v2':
            raise ValueError(f'{rel}: candidate {view["run_id"]} must be a FreqCone v2 run')
        if fc.get('aperture') not in APERTURE_RANK:
            raise ValueError(f'{rel}: aperture {fc.get("aperture")!r} is not one of A1-A3')
        val = view.get('val') or {}
        if val.get('metric') != VAL_METRIC:
            raise ValueError(f'{rel}: validation metric {val.get("metric")!r}, expected {VAL_METRIC}')
        view['_file'] = rel
        views.append(view)
        hashes[rel] = digest
    ids = [v['run_id'] for v in views]
    if len(set(ids)) != len(ids):
        raise ValueError(f'duplicate run ids among the candidates: {ids}')
    if not views:
        raise ValueError(f'no candidate run ({", ".join(CANDIDATES)}) in {runs_root}')
    ref = views[0]
    for v in views[1:]:
        if v['val']['val_list_sha256'] != ref['val']['val_list_sha256'] or \
                v['val']['per_sample_true'] != ref['val']['per_sample_true']:
            raise ValueError(f"{v['_file']}: validation images differ from {ref['_file']}")
        if (v['dataset'], v['seed']) != (ref['dataset'], ref['seed']):
            raise ValueError(f"{v['_file']}: dataset / seed differ from {ref['_file']}")
    return views, hashes


def simplicity(view):
    fc = view['freqcone']
    return (fc['w_ent'], APERTURE_RANK[fc['aperture']])


def stratified_indices(y_true, n_boot, seed):
    y = np.asarray(y_true)
    rng = np.random.default_rng(seed)
    groups = [np.nonzero(y == c)[0] for c in np.unique(y)]
    return np.stack([np.concatenate([rng.choice(g, size=g.size, replace=True) for g in groups])
                     for _ in range(n_boot)])


def apply_rule(views, scores, boot):
    """scores: run_id -> s_m; boot: run_id -> array of s_m(b). Returns m*, the table and the choice."""
    order = sorted(views, key=simplicity)                     # argmax ties go to the simplest run
    m_star = max(order, key=lambda v: scores[v['run_id']])['run_id']
    rows = []
    for v in order:
        rid = v['run_id']
        diff = boot[m_star] - boot[rid]
        se = float(np.std(diff, ddof=1))
        gap = scores[m_star] - scores[rid]
        rows.append({'run_id': rid, 'w_ent': v['freqcone']['w_ent'], 'aperture': v['freqcone']['aperture'],
                     's': scores[rid], 'gap_to_best': gap, 'SE': se, 'tied': bool(gap <= se + 1e-12)})
    tied = [row for row in rows if row['tied']]
    chosen = min(tied, key=lambda row: (row['w_ent'], APERTURE_RANK[row['aperture']]))
    return m_star, rows, chosen


def select(runs_root, n_boot=B_DEFAULT, seed=SEED_DEFAULT):
    views, hashes = load_candidates(runs_root)
    y_true = np.asarray(views[0]['val']['per_sample_true'], dtype=int)
    idx = stratified_indices(y_true, n_boot, seed)
    scores, boot = {}, {}
    for v in views:
        pred = np.asarray(v['val']['per_sample_pred'], dtype=int)
        s = hungarian_acc(y_true, pred)
        if abs(s - v['val']['value']) > 1e-9:
            raise ValueError(f"{v['_file']}: stored val.value {v['val']['value']} != recomputed {s}")
        scores[v['run_id']] = s
        boot[v['run_id']] = np.array([hungarian_acc(y_true[i], pred[i]) for i in idx])
    m_star, rows, chosen = apply_rule(views, scores, boot)
    return {
        'rule': __doc__.split('usage:')[0].strip(), 'created_utc': datetime.now(timezone.utc).isoformat(),
        'dataset': views[0]['dataset'], 'seed': views[0]['seed'],
        'bootstrap': {'B': n_boot, 'seed': seed, 'stratified_by_class': True, 'paired': True},
        'val_metric': VAL_METRIC, 'val_list_sha256': views[0]['val']['val_list_sha256'], 'n_val': int(y_true.size),
        'candidates_present': [v['run_id'] for v in sorted(views, key=simplicity)],
        'candidates_missing': [c for c in CANDIDATES if c not in scores],
        'm_star': m_star, 'table': rows, 'selected': chosen, 'inputs_sha256': hashes,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description='pre-registered pilot selection (validation fields only)')
    ap.add_argument('runs_root')
    ap.add_argument('--out', default=None, help='default: RUNS_ROOT/selection.json')
    ap.add_argument('--n_boot', type=int, default=B_DEFAULT)
    ap.add_argument('--seed', type=int, default=SEED_DEFAULT)
    a = ap.parse_args(argv)
    out = select(a.runs_root, a.n_boot, a.seed)
    path = a.out or os.path.join(a.runs_root, 'selection.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print(f"{'run':5s} {'w_ent':>6s} {'ap':>3s} {'val':>6s} {'gap':>6s} {'SE':>6s} tied", file=sys.stderr)
    for r in out['table']:
        print(f"{r['run_id']:5s} {r['w_ent']:6g} {r['aperture']:>3s} {100 * r['s']:6.2f} {100 * r['gap_to_best']:6.2f} "
              f"{100 * r['SE']:6.2f} {'yes' if r['tied'] else ''}", file=sys.stderr)
    sel = out['selected']
    print(f"selected {sel['run_id']} (lambda_ent {sel['w_ent']}, {sel['aperture']}); m* = {out['m_star']}; "
          f"missing: {out['candidates_missing'] or 'none'}; written to {path}", file=sys.stderr)
    print(f"{sel['w_ent']} {sel['aperture']}")
    return out


if __name__ == '__main__':
    main()
