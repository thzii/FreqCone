"""Gates and checks of the Stanford Cars plan: resolution / level rule, G0, budget overrun, preflight."""
import json
import os
import subprocess

import pytest

from freqcone import pilot_tools as pt
from freqcone.hypcd import SCHEMA
from helpers import write_run


def probe(root, tag, est, batch=128):
    os.makedirs(os.path.join(root, tag), exist_ok=True)
    rec = {'schema': SCHEMA, 'run_tag': tag, 'batch_size': batch, 'profile': {'est_epoch_s': est},
           'cost': {'peak_mem_gb': 10.0}}
    with open(os.path.join(root, tag, 'run.json'), 'w') as f:
        json.dump(rec, f)


def test_resolution_and_level(tmp_path):
    probe(tmp_path, 'lowmem', 100.0)
    probe(tmp_path, 'fc224', 175.0)
    probe(tmp_path, 'fc112', 120.0)
    d = pt.decide(str(tmp_path))
    assert d['resolution'] == 224 and d['coarse_res_flag'] == 0              # 1.75 x <= 1.8 x
    assert d['level'] == 'A'                                                  # 200 x 175 s = 9.7 h <= 10 h
    assert d['gpu_hours']['A'] == pytest.approx(200 * (4 * 100 + 6 * 175) / 3600)
    assert d['gpu_hours']['B'] == pytest.approx(200 * (3 * 100 + 3 * 175) / 3600)
    assert set(d['planned_hours']) == {'R0', 'R1', 'R2', 'R3', 'base_s8', 'base_s9', 'base_s10', 'fc_s8', 'fc_s9',
                                       'fc_s10', 'R0-repro'}
    probe(tmp_path, 'fc224', 185.0)                                           # 1.85 x: too slow -> 112 px
    d = pt.decide(str(tmp_path))
    assert d['resolution'] == 112 and d['coarse_res_flag'] == 112 and d['level'] == 'A'
    probe(tmp_path, 'fc224', 150.0, batch=64)                                 # only fits a smaller batch -> 112 px
    assert pt.decide(str(tmp_path))['resolution'] == 112
    probe(tmp_path, 'lowmem', 150.0)
    probe(tmp_path, 'fc224', 200.0)                                           # 200 x 200 s = 11.1 h > 10 h -> level B
    d = pt.decide(str(tmp_path))
    assert d['level'] == 'B' and 'R1' not in d['planned_hours'] and 'fc_s8' in d['planned_hours']


def test_decision_requires_the_probes(tmp_path):
    with pytest.raises(RuntimeError, match='baseline'):
        pt.decide(str(tmp_path))
    probe(tmp_path, 'lowmem', 100.0)
    with pytest.raises(RuntimeError, match='neither'):
        pt.decide(str(tmp_path))


def test_readme_section_is_replaced(tmp_path):
    readme = tmp_path / 'README.md'
    readme.write_text(f'head\n{pt.README_START}\nold\n{pt.README_END}\ntail\n', encoding='utf-8')
    probe(tmp_path, 'lowmem', 100.0)
    probe(tmp_path, 'fc224', 150.0)
    pt.write_readme(pt.decide(str(tmp_path)), readme=str(readme))
    text = readme.read_text(encoding='utf-8')
    assert 'old' not in text and '**224 px**' in text and '**Mức A**' in text and text.endswith('tail\n')


def g0_run(root, run_id='R0', all_at_up=0.830, enable=False, **over):
    d = write_run(root, run_id, over.pop('seed', 7), enable,
                  {'all': [0.70, all_at_up, 0.80], 'old': [0.9, 0.93, 0.92], 'new': [0.6, 0.78, 0.74]},
                  val_acc=[0.5, 0.6, 0.7], disjoint_old=[0.80, 0.95, 0.90])
    rec = json.loads(open(os.path.join(d, 'run.json')).read())
    rec.update(over)
    with open(os.path.join(d, 'run.json'), 'w') as f:
        json.dump(rec, f)
    return d


def test_g0(tmp_path):
    assert pt.g0(g0_run(tmp_path / 'a', all_at_up=0.830))['passed']          # row (1), epoch 2 (upstream rule): 83.0 vs 83.8
    assert pt.g0(g0_run(tmp_path / 'b', all_at_up=0.858))['passed']
    assert not pt.g0(g0_run(tmp_path / 'c', all_at_up=0.817))['passed']      # -2.1
    out = pt.g0(g0_run(tmp_path / 'd', all_at_up=0.836))
    assert out['gcd_row']['epoch'] == 2 and out['last_row']['epoch'] == 3
    assert not pt.g0(g0_run(tmp_path / 'e', enable=True))['passed']         # G0 is about the baseline
    assert not pt.g0(g0_run(tmp_path / 'f', c=0.05, r=2.3))['passed']
    repro = g0_run(tmp_path / 'g', run_id='R0-repro', fix_label_order=False, split_train_val=False, seed=1)
    assert pt.g0(repro)['passed']
    bad = g0_run(tmp_path / 'h', run_id='R0-repro', fix_label_order=True, split_train_val=False, seed=1)
    assert not pt.g0(bad)['passed']


def test_budget_overrun(tmp_path):
    decision = {'planned_hours': {'R0': 10.0, 'R1': 10.0, 'R2': 15.0}}
    for rid, hours in (('R0', 12.0), ('R1', 12.5)):
        os.makedirs(tmp_path / rid)
        (tmp_path / rid / 'run.json').write_text(json.dumps({'cost': {'wall_s': hours * 3600}}))
    assert not pt.budget(str(tmp_path), decision)['over']                     # 24.5 h for 20 h: within 30 %
    (tmp_path / 'R1' / 'run.json').write_text(json.dumps({'cost': {'wall_s': 15 * 3600}}))
    assert pt.budget(str(tmp_path), decision)['over']                         # 27 h > 26 h


def git(repo, *args):
    subprocess.run(['git', *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / 'repo'
    root.mkdir()
    git(root, 'init', '-q')
    git(root, 'config', 'user.email', 'test@example.com')
    git(root, 'config', 'user.name', 'test')
    (root / 'config.py').write_text('car_root = "{DATA_PATH}"\n')
    (root / 'cutoffs.json').write_text('{}')
    os.makedirs(root / 'runs' / 'profile_scars')
    (root / 'runs' / 'profile_scars' / 'decision.json').write_text('{}')
    git(root, 'add', '-A')
    git(root, 'commit', '-q', '-m', 'init')
    git(root, 'tag', 'pilot-v2')
    return root


def test_preflight(repo, monkeypatch):
    monkeypatch.setattr(pt, 'REPO_ROOT', str(repo))
    monkeypatch.chdir(repo)
    runs, decision, smoke = 'runs', 'runs/profile_scars/decision.json', 'runs/smoke_scars'
    (repo / 'config.py').write_text('car_root = "/data/cars"\n')                # filled-in paths are allowed
    assert pt.preflight(runs, decision, smoke) == []
    os.makedirs(repo / smoke)
    assert any('smoke' in p for p in pt.preflight(runs, decision, smoke))
    os.rmdir(repo / smoke)
    (repo / 'cutoffs.json').write_text('{"scars": 1}')
    assert any('cutoffs.json' in p for p in pt.preflight(runs, decision, smoke))
    git(repo, 'checkout', '-q', '--', 'cutoffs.json')
    os.makedirs(repo / 'runs' / 'R0')
    (repo / 'runs' / 'R0' / 'run.json').write_text('{}')
    (repo / 'runs' / 'selection.json').write_text('{}')
    git(repo, 'add', 'runs')
    git(repo, 'commit', '-q', '-m', 'results')                                  # results after the tag: allowed
    assert pt.preflight(runs, decision, smoke) == []
    (repo / 'other.py').write_text('x = 1\n')
    git(repo, 'add', 'other.py')
    git(repo, 'commit', '-q', '-m', 'code change')
    assert any('differs from tag' in p for p in pt.preflight(runs, decision, smoke))
