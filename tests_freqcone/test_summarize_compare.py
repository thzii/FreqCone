"""Plan 2.2-2.3: one run's table (two main rows, secondary row, oracle rows) and the paired comparison with gates."""
import json
import os
import subprocess

import numpy as np
import pytest

from freqcone import compare_runs as cr
from freqcone import summarize_run as sr
from freqcone.runs import LockError, check_locked
from helpers import write_run


def curves(all_, old, new):
    return {'all': list(all_), 'old': list(old), 'new': list(new)}


def test_rows_pick_the_right_epochs(tmp_path):
    d = write_run(tmp_path, 'R0', 7, False,
                  curves([.70, .80, .78, .79], [.90, .88, .95, .93], [.60, .75, .70, .72]),
                  val_acc=[.5, .7, .7, .6], disjoint_old=[.10, .30, .30, .20])
    out, md = sr.summarize(d)
    rows = {r['key']: r for r in out['rows']}
    assert rows['last']['epoch'] == 4 and rows['last']['all'] == .79
    assert rows['best_val']['epoch'] == 3                     # tie on validation: the later epoch
    assert rows['gcd']['epoch'] == 2 and rows['gcd']['all'] == .80   # tie on disjoint-test Old: the first epoch (strict '>')
    assert rows['gcd']['source'] == 'ss_kmeans_epoch_start' and rows['gcd']['weights_after_epoch'] == 1
    assert (rows['oracle_all']['epoch'], rows['oracle_old']['epoch'], rows['oracle_new']['epoch']) == (2, 3, 2)
    assert all(rows[k]['oracle'] for k in ('oracle_all', 'oracle_old', 'oracle_new')) and not rows['last']['oracle']
    assert 'Chuẩn GCD (theo code HypCD)' in md and 'Epoch cuối' in md and 'Oracle (chọn bằng test' in md


def test_confirmation_runs_have_no_best_validation_row(tmp_path):
    d = write_run(tmp_path, 'B8', 8, False, curves([.70, .80], [.90, .88], [.60, .75]), split_train_val=False)
    out, md = sr.summarize(d)
    assert [r['key'] for r in out['rows']] == ['gcd', 'last', 'oracle_all', 'oracle_old', 'oracle_new']
    assert 'val cao nhất' not in md and 'Chuẩn GCD (theo code HypCD)' in md and 'Epoch cuối' in md and 'Oracle' in md


def git(repo, *args):
    subprocess.run(['git', *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / 'repo'
    root.mkdir()
    git(root, 'init', '-q')
    git(root, 'config', 'user.email', 'test@example.com')
    git(root, 'config', 'user.name', 'test')
    (root / 'README.md').write_text('x\n')
    git(root, 'add', '-A')
    git(root, 'commit', '-q', '-m', 'init')
    return root


def test_candidates_are_locked_until_the_selection_is_committed(repo, monkeypatch):
    runs = repo / 'runs'
    d = write_run(runs, 'R1', 7, True, curves([.8], [.9], [.7]), val_acc=[.6])
    import freqcone.runs as runs_mod
    monkeypatch.setattr(runs_mod, 'REPO_ROOT', str(repo))
    sel = {'inputs_sha256': {'R1/run.json': __import__('hashlib').sha256((runs / 'R1' / 'run.json').read_bytes()).hexdigest()}}
    (runs / 'selection.json').write_text(json.dumps(sel))
    with pytest.raises(LockError, match='not committed'):
        runs_mod.load_run(d, with_test=True, repo=str(repo))
    git(repo, 'add', 'runs/selection.json')
    git(repo, 'commit', '-q', '-m', 'lock')
    assert runs_mod.load_run(d, with_test=True, repo=str(repo))['test'][0]['eval']['all'] == .8
    (runs / 'R1' / 'run.json').write_text(json.dumps({**json.loads((runs / 'R1' / 'run.json').read_text()), 'x': 1}))
    with pytest.raises(LockError, match='changed after the selection'):
        check_locked(str(runs / 'selection.json'), str(repo))
    d0 = write_run(runs, 'R0', 7, False, curves([.8], [.9], [.7]))
    assert runs_mod.load_run(d0, with_test=True, repo=str(repo))['test']          # baselines are never locked


def arm(root, prefix, enable, new_shift, all_shift=0.0, epochs=5):
    """Confirmation runs: no validation split, so no validation accuracy."""
    dirs = []
    for s in (8, 9, 10):
        rng = np.random.default_rng(s)
        base = np.linspace(.70, .80, epochs) + rng.normal(0, .002, epochs)
        c = curves(base + all_shift, base + .1, base - .05 + new_shift[s])
        dirs.append(write_run(root, f'{prefix}{s}', s, enable, c, split_train_val=False))
    return dirs


def test_final_gate_and_tables(tmp_path):
    base = arm(tmp_path, 'B', False, {8: 0, 9: 0, 10: 0})
    fc = arm(tmp_path, 'F', True, {8: .02, 9: .015, 10: -.005})          # +2.0, +1.5, -0.5 points on New
    pairs = cr.pair_up(base, fc)
    cr.check_gate_runs(pairs, 'final')
    d = cr.deltas(pairs)
    assert d['best_val']['per_seed'] == []                               # no such row without a validation split
    last = d['last']['stats']
    assert last['new']['mean'] == pytest.approx(1.0, abs=1e-9) and last['new']['wins'] == 2
    g = cr.gate(d, 'final')
    assert g['pass']                                                     # mean +1.0, 2/3 wins, dAll 0
    md = cr.markdown(d, g)
    assert 'Oracle (chọn bằng test, tham khảo)' in md and md.index('Oracle') > md.index('Epoch cuối')
    assert 'val cao nhất' not in md
    fc_bad = arm(tmp_path, 'Fb', True, {8: .04, 9: -.001, 10: -.001})    # mean +1.27 but one win only
    assert not cr.gate(cr.deltas(cr.pair_up(base, fc_bad)), 'final')['pass']
    fc_all = arm(tmp_path, 'Fa', True, {8: .02, 9: .02, 10: .02}, all_shift=-.006)
    assert not cr.gate(cr.deltas(cr.pair_up(base, fc_all)), 'final')['pass']    # All drops by 0.6


def test_g1_gate(tmp_path):
    # non-candidate ids, so that no selection lock is involved here (the lock has its own test)
    b = write_run(tmp_path, 'base_p', 7, False, curves([.800], [.90], [.780]), val_acc=[.6])
    f = write_run(tmp_path, 'fc_p', 7, True, curves([.796], [.89], [.790]), val_acc=[.6])
    cr.check_gate_runs(cr.pair_up([b], [f]), 'g1')
    d = cr.deltas(cr.pair_up([b], [f]))
    assert cr.gate(d, 'g1')['pass']                                               # dNew +1.0, dAll -0.4
    assert 'val cao nhất' in cr.markdown(d)                                       # pilot runs keep this row
    f2 = write_run(tmp_path, 'fc_q', 7, True, curves([.794], [.89], [.795]))
    assert not cr.gate(cr.deltas(cr.pair_up([b], [f2])), 'g1')['pass']            # dAll -0.6
    f3 = write_run(tmp_path, 'fc_r', 7, True, curves([.800], [.89], [.789]))
    assert not cr.gate(cr.deltas(cr.pair_up([b], [f3])), 'g1')['pass']            # dNew +0.9


def test_gates_refuse_the_wrong_data_mode(tmp_path):
    with_val = {'base': write_run(tmp_path, 'a', 7, False, curves([.8], [.9], [.7]), val_acc=[.6]),
                'fc': write_run(tmp_path, 'b', 7, True, curves([.8], [.9], [.7]), val_acc=[.6])}
    without = {'base': write_run(tmp_path, 'c', 7, False, curves([.8], [.9], [.7]), split_train_val=False),
               'fc': write_run(tmp_path, 'd', 7, True, curves([.8], [.9], [.7]), split_train_val=False)}
    with pytest.raises(ValueError, match='gate final expects runs without'):
        cr.check_gate_runs(cr.pair_up([with_val['base']], [with_val['fc']]), 'final')
    with pytest.raises(ValueError, match='gate g1 expects runs with'):
        cr.check_gate_runs(cr.pair_up([without['base']], [without['fc']]), 'g1')
    with pytest.raises(ValueError, match='one arm holds out the validation split'):
        cr.pair_up([with_val['base']], [without['fc']])
    cr.check_gate_runs(cr.pair_up([with_val['base']], [with_val['fc']]), 'none')


def test_plot_is_written(tmp_path):
    base = arm(tmp_path, 'B', False, {8: 0, 9: 0, 10: 0})
    fc = arm(tmp_path, 'F', True, {8: .02, 9: .01, 10: .0})
    path = cr.plot(cr.pair_up(base, fc), str(tmp_path / 'curves.png'))
    assert os.path.getsize(path) > 10_000
    with pytest.raises(ValueError, match='seeds differ'):
        cr.pair_up(base[:2], fc)
