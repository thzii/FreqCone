"""Plan 2.2-2.3: one run's table (main, secondary, upstream, oracle rows) and the paired comparison with gates."""
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
                  val_acc=[.5, .7, .7, .6], upstream_old=[.10, .30, .30, .20])
    out, md = sr.summarize(d)
    rows = {r['key']: r for r in out['rows']}
    assert rows['last']['epoch'] == 4 and rows['last']['all'] == .79
    assert rows['best_val']['epoch'] == 3                     # tie on validation: the later epoch
    assert rows['upstream']['epoch'] == 2                     # tie on upstream Old: the first epoch (strict '>')
    assert (rows['oracle_all']['epoch'], rows['oracle_old']['epoch'], rows['oracle_new']['epoch']) == (2, 3, 2)
    assert all(rows[k]['oracle'] for k in ('oracle_all', 'oracle_old', 'oracle_new')) and not rows['last']['oracle']
    assert 'KẾT QUẢ CHÍNH' in md and 'thổi phồng' in md


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
    assert runs_mod.load_run(d, with_test=True, repo=str(repo))['test'][0]['all'] == .8
    (runs / 'R1' / 'run.json').write_text(json.dumps({**json.loads((runs / 'R1' / 'run.json').read_text()), 'x': 1}))
    with pytest.raises(LockError, match='changed after the selection'):
        check_locked(str(runs / 'selection.json'), str(repo))
    d0 = write_run(runs, 'R0', 7, False, curves([.8], [.9], [.7]))
    assert runs_mod.load_run(d0, with_test=True, repo=str(repo))['test']          # baselines are never locked


def arm(root, prefix, enable, new_shift, all_shift=0.0, epochs=5):
    dirs = []
    for s in (8, 9, 10):
        rng = np.random.default_rng(s)
        base = np.linspace(.70, .80, epochs) + rng.normal(0, .002, epochs)
        c = curves(base + all_shift, base + .1, base - .05 + new_shift[s])
        dirs.append(write_run(root, f'{prefix}_s{s}', s, enable, c, val_acc=list(base)))
    return dirs


def test_final_gate_and_tables(tmp_path):
    base = arm(tmp_path, 'base', False, {8: 0, 9: 0, 10: 0})
    fc = arm(tmp_path, 'fc', True, {8: .02, 9: .015, 10: -.005})        # +2.0, +1.5, -0.5 points on New
    pairs = cr.pair_up(base, fc)
    d = cr.deltas(pairs)
    last = d['last']['stats']
    assert last['new']['mean'] == pytest.approx(1.0, abs=1e-9) and last['new']['wins'] == 2
    g = cr.gate(d, 'final')
    assert g['pass']                                                     # mean +1.0, 2/3 wins, dAll 0
    md = cr.markdown(d, g)
    assert 'Oracle (chọn bằng test, tham khảo)' in md and md.index('Oracle') > md.index('Epoch cuối')
    fc_bad = arm(tmp_path, 'fcb', True, {8: .04, 9: -.001, 10: -.001})  # mean +1.27 but one win only
    assert not cr.gate(cr.deltas(cr.pair_up(base, fc_bad)), 'final')['pass']
    fc_all = arm(tmp_path, 'fca', True, {8: .02, 9: .02, 10: .02}, all_shift=-.006)
    assert not cr.gate(cr.deltas(cr.pair_up(base, fc_all)), 'final')['pass']    # All drops by 0.6


def test_g1_gate(tmp_path):
    # non-candidate ids, so that no selection lock is involved here (the lock has its own test)
    b = write_run(tmp_path, 'base_p', 7, False, curves([.800], [.90], [.780]))
    f = write_run(tmp_path, 'fc_p', 7, True, curves([.796], [.89], [.790]))
    assert cr.gate(cr.deltas(cr.pair_up([b], [f])), 'g1')['pass']                 # dNew +1.0, dAll -0.4
    f2 = write_run(tmp_path, 'fc_q', 7, True, curves([.794], [.89], [.795]))
    assert not cr.gate(cr.deltas(cr.pair_up([b], [f2])), 'g1')['pass']            # dAll -0.6
    f3 = write_run(tmp_path, 'fc_r', 7, True, curves([.800], [.89], [.789]))
    assert not cr.gate(cr.deltas(cr.pair_up([b], [f3])), 'g1')['pass']            # dNew +0.9


def test_plot_is_written(tmp_path):
    base = arm(tmp_path, 'base', False, {8: 0, 9: 0, 10: 0})
    fc = arm(tmp_path, 'fc', True, {8: .02, 9: .01, 10: .0})
    path = cr.plot(cr.pair_up(base, fc), str(tmp_path / 'curves.png'))
    assert os.path.getsize(path) > 10_000
    with pytest.raises(ValueError, match='seeds differ'):
        cr.pair_up(base[:2], fc)
