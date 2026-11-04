"""Paired, class-stratified bootstrap 1-SE rule on the Stanford Cars pilot candidates R1-R3."""
import hashlib
import json
import os

import numpy as np
import pytest

from freqcone import select_pilot as sp
from freqcone.hypcd import SCHEMA, VAL_METRIC

N_CLASSES, PER_CLASS = 98, 4
TRUE = np.repeat(np.arange(N_CLASSES), PER_CLASS)
CONFIG = {'R1': (0.03, 'A1'), 'R2': (0.1, 'A1'), 'R3': (0.1, 'A3')}


def preds(acc, seed):
    """Exactly round((1 - acc) * n) wrong predictions, at seed-dependent positions, each to a random other class."""
    rng = np.random.default_rng(seed)
    wrong = rng.permutation(TRUE.size)[:int(round((1 - acc) * TRUE.size))]
    out = TRUE.copy()
    out[wrong] = (TRUE[wrong] + rng.integers(1, N_CLASSES, wrong.size)) % N_CLASSES
    return out


def write(root, rid, pred, variant='v2', sha='val-a', enable=True, w_ent=None, aperture=None):
    from project_utils.cluster_utils import cluster_acc
    w, ap = CONFIG.get(rid, (0.03, 'A1'))
    rec = {'schema': SCHEMA, 'run_id': rid, 'dataset': 'scars', 'seed': 7,
           'freqcone': {'enable': enable, 'variant': variant if enable else None, 'w_ent': w_ent or w,
                        'aperture': aperture or ap},
           'val': {'metric': VAL_METRIC, 'value': float(cluster_acc(TRUE, pred)), 'per_sample_true': TRUE.tolist(),
                   'per_sample_pred': [int(x) for x in pred], 'val_list_sha256': sha}}
    os.makedirs(os.path.join(root, rid), exist_ok=True)
    with open(os.path.join(root, rid, 'run.json'), 'w') as f:
        json.dump(rec, f)


def grid(root, accs, n_boot=300):
    write(root, 'R0', preds(0.8, 99), enable=False)
    for i, rid in enumerate(('R1', 'R2', 'R3')):
        write(root, rid, preds(accs[rid], i))
    return sp.select(str(root), n_boot=n_boot)


def test_clear_winner(tmp_path):
    out = grid(tmp_path, {'R1': .5, 'R2': .95, 'R3': .5})
    assert out['m_star'] == 'R2' and out['selected']['run_id'] == 'R2'
    assert [r['run_id'] for r in out['table'] if r['tied']] == ['R2'] and 'R0' not in out['candidates_present']


def test_ties_go_to_smaller_lambda_then_wider_cone(tmp_path):
    out = grid(tmp_path, {'R1': .80, 'R2': .80, 'R3': .80})            # gaps well inside one SE (~3 points)
    assert {r['run_id'] for r in out['table'] if r['tied']} == {'R1', 'R2', 'R3'}
    assert out['selected']['run_id'] == 'R1'                           # lambda 0.03 first
    out = grid(tmp_path, {'R1': .40, 'R2': .80, 'R3': .80})
    assert out['selected']['run_id'] == 'R3'                           # same lambda: A3 (wider) before A1
    se = {r['run_id']: r['SE'] for r in out['table']}
    assert 0.01 < se['R3' if out['m_star'] == 'R2' else 'R2'] < 0.06


def test_bootstrap_is_stratified_paired_and_reproducible(tmp_path):
    idx = sp.stratified_indices(TRUE, 50, 0)
    for b in idx:
        assert np.array_equal(np.bincount(TRUE[b], minlength=N_CLASSES), np.full(N_CLASSES, PER_CLASS))
    same = preds(0.8, 5)
    write(tmp_path, 'R1', same)
    write(tmp_path, 'R3', same)
    out = sp.select(str(tmp_path), n_boot=200)
    assert all(r['SE'] == 0.0 and r['tied'] for r in out['table'])     # paired: identical runs never differ
    assert out['selected']['run_id'] == 'R1' and out['candidates_missing'] == ['R2']
    assert sp.select(str(tmp_path), n_boot=200)['table'] == out['table']


def test_inputs_are_hashed_and_only_validation_is_read(tmp_path):
    out = grid(tmp_path, {'R1': .6, 'R2': .5, 'R3': .5}, n_boot=50)
    assert set(out['inputs_sha256']) == {'R1/run.json', 'R2/run.json', 'R3/run.json'}
    for name, digest in out['inputs_sha256'].items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest
    assert set(sp.VAL_FIELDS) == {'schema', 'run_id', 'dataset', 'seed', 'freqcone', 'val'}


def test_inconsistent_inputs_are_refused(tmp_path):
    write(tmp_path, 'R1', preds(.8, 1))
    write(tmp_path, 'R2', preds(.8, 2), sha='val-b')
    with pytest.raises(ValueError, match='validation images differ'):
        sp.select(str(tmp_path), n_boot=10)
    write(tmp_path, 'R2', preds(.8, 2), variant='v3')
    with pytest.raises(ValueError, match='v2'):
        sp.select(str(tmp_path), n_boot=10)


def test_cli_prints_the_selected_configuration(tmp_path, capsys):
    grid(tmp_path, {'R1': .5, 'R2': .5, 'R3': .95}, n_boot=20)
    sp.main([str(tmp_path), '--n_boot', '50'])
    assert capsys.readouterr().out.strip().splitlines()[-1] == '0.1 A3'
    assert (tmp_path / 'selection.json').exists()
