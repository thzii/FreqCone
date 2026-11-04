"""Section 1.2: paired, class-stratified bootstrap 1-SE rule, on synthetic run records."""
import hashlib
import json

import numpy as np
import pytest

from freqcone import select_pilot as sp
from freqcone.hypcd import SCHEMA, VAL_METRIC

N_CLASSES, PER_CLASS = 100, 3
TRUE = np.repeat(np.arange(N_CLASSES), PER_CLASS)


def preds(acc, seed):
    """Exactly round((1 - acc) * n) wrong predictions, at seed-dependent positions, each to a random other class."""
    rng = np.random.default_rng(seed)
    wrong = rng.permutation(TRUE.size)[:int(round((1 - acc) * TRUE.size))]
    out = TRUE.copy()
    out[wrong] = (TRUE[wrong] + rng.integers(1, N_CLASSES, wrong.size)) % N_CLASSES
    return out


def record(rid, w_ent, aperture, pred, variant='v2', sha='val-a', enable=True):
    from project_utils.cluster_utils import cluster_acc
    return {'schema': SCHEMA, 'run_id': rid, 'dataset': 'cub', 'seed': 7,
            'freqcone': {'enable': enable, 'variant': variant if enable else None, 'w_ent': w_ent, 'aperture': aperture},
            'val': {'metric': VAL_METRIC, 'value': float(cluster_acc(TRUE, pred)), 'per_sample_true': TRUE.tolist(),
                    'per_sample_pred': [int(x) for x in pred], 'val_list_sha256': sha},
            'test_last': {'all': 0.123456789, 'old': 0.2, 'new': 0.3}}       # sentinel: must never be read


def write(tmp_path, recs):
    for r in recs:
        (tmp_path / f"{r['run_id']}.json").write_text(json.dumps(r))


def grid(tmp_path, accs, n_boot=300):
    recs = [record('R0', 0.0, 'A1', preds(0.80, 99), enable=False)]
    configs = {'R1': (0.0, 'A1'), 'R2': (0.03, 'A1'), 'R3': (0.03, 'A2'), 'R4': (0.03, 'A3'),
               'R5': (0.1, 'A1'), 'R6': (0.1, 'A2'), 'R7': (0.1, 'A3')}
    for i, (rid, (w, ap)) in enumerate(configs.items()):
        recs.append(record(rid, w, ap, preds(accs[rid], i)))
    write(tmp_path, recs)
    return sp.select(str(tmp_path), n_boot=n_boot)


def test_clear_winner_is_selected_and_test_fields_are_ignored(tmp_path):
    out = grid(tmp_path, {'R1': .5, 'R2': .5, 'R3': .5, 'R4': .5, 'R5': .95, 'R6': .5, 'R7': .5})
    assert out['m_star'] == 'R5' and out['selected']['run_id'] == 'R5'
    assert [r['run_id'] for r in out['table'] if r['tied']] == ['R5']
    assert '0.123456789' not in json.dumps(out) and 'R0' not in out['candidates_present']
    assert out['r8_config'] == {**out['r8_config'], 'w_ent': 0.1, 'aperture': 'A1'}


def test_ties_go_to_smaller_lambda_then_wider_cone(tmp_path):
    # R2, R4, R5: 80 % with different error sets (gaps well inside one SE, about 3 points at n = 300);
    # R1 30 %, R3 / R6 / R7 60 %: far outside
    out = grid(tmp_path, {'R1': .30, 'R2': .80, 'R3': .60, 'R4': .80, 'R5': .80, 'R6': .60, 'R7': .60})
    tied = {r['run_id'] for r in out['table'] if r['tied']}
    assert tied == {'R2', 'R4', 'R5'}
    se = {r['run_id']: r['SE'] for r in out['table']}
    assert all(0.015 < se[r] < 0.06 for r in tied - {out['m_star']})
    assert out['selected']['run_id'] == 'R4'                             # lambda 0.03 first, then A3 < A2 < A1
    assert out['r8_config']['from_run'] == 'R4'


def test_r8_excludes_lambda_zero(tmp_path):
    out = grid(tmp_path, {'R1': .95, 'R2': .5, 'R3': .5, 'R4': .5, 'R5': .6, 'R6': .5, 'R7': .5})
    assert out['selected']['run_id'] == 'R1'                             # reported as is: L_dir + L_con only
    assert out['r8_config']['from_run'] == 'R5' and out['r8_config']['w_ent'] == 0.1


def test_bootstrap_is_stratified_paired_and_reproducible(tmp_path):
    idx = sp.stratified_indices(TRUE, 50, 0)
    for b in idx:
        assert np.array_equal(np.bincount(TRUE[b], minlength=N_CLASSES), np.full(N_CLASSES, PER_CLASS))
    assert np.array_equal(idx, sp.stratified_indices(TRUE, 50, 0))
    same = preds(0.8, 5)
    write(tmp_path, [record('R2', 0.03, 'A1', same), record('R4', 0.03, 'A3', same)])
    out = sp.select(str(tmp_path), n_boot=200)
    assert all(r['SE'] == 0.0 and r['tied'] for r in out['table'])     # paired: identical runs never differ
    assert out['selected']['run_id'] == 'R4'
    assert sp.select(str(tmp_path), n_boot=200)['table'] == out['table']


def test_inputs_are_hashed(tmp_path):
    out = grid(tmp_path, {'R1': .5, 'R2': .6, 'R3': .5, 'R4': .5, 'R5': .5, 'R6': .5, 'R7': .5}, n_boot=50)
    for name, digest in out['inputs_sha256'].items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest
    assert set(out['inputs_sha256']) == {f'R{i}.json' for i in range(1, 8)}


def test_inconsistent_inputs_are_refused(tmp_path):
    write(tmp_path, [record('R2', 0.03, 'A1', preds(.8, 1)), record('R4', 0.03, 'A3', preds(.8, 2), sha='val-b')])
    with pytest.raises(ValueError, match='validation images differ'):
        sp.select(str(tmp_path), n_boot=10)
    (tmp_path / 'R4.json').write_text(json.dumps(record('R4', 0.03, 'A3', preds(.8, 2), variant='v3')))
    with pytest.raises(ValueError, match='v2'):
        sp.select(str(tmp_path), n_boot=10)


def test_missing_a2_runs_are_listed(tmp_path):
    recs = [record(rid, w, ap, preds(.7, i)) for i, (rid, w, ap) in
            enumerate([('R1', 0.0, 'A1'), ('R2', 0.03, 'A1'), ('R4', 0.03, 'A3'), ('R5', 0.1, 'A1'), ('R7', 0.1, 'A3')])]
    write(tmp_path, recs)
    out = sp.select(str(tmp_path), n_boot=50)
    assert out['candidates_missing'] == ['R3', 'R6']
