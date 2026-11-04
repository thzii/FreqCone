"""Gates and checks of the pilot: resolution / budget rule, G0, budget overrun, preflight, report lock, G1."""
import json
import os
import subprocess

import pytest

from freqcone import pilot_tools as pt
from freqcone import report_pilot as rp
from freqcone.hypcd import SCHEMA


def probe(tmp_path, tag, est, batch=128):
    rec = {'schema': SCHEMA, 'run_tag': tag, 'batch_size': batch, 'profile': {'est_epoch_s': est},
           'cost': {'peak_mem_gb': 10.0}}
    (tmp_path / f'{tag}.json').write_text(json.dumps(rec))


def test_resolution_rule_and_budget(tmp_path):
    probe(tmp_path, 'lowmem', 100.0)
    probe(tmp_path, 'fc224', 175.0)
    probe(tmp_path, 'fc112', 120.0)
    d = pt.decide(str(tmp_path))
    assert d['resolution'] == 224 and d['coarse_size_flag'] == 0             # 1.75 x <= 1.8 x
    assert abs(d['gpu_hours_all_runs'] - 200 * (2 * 100 + 8 * 175) / 3600) < 1e-9 and not d['drop_A2']
    probe(tmp_path, 'fc224', 185.0)                                             # 1.85 x: too slow
    d = pt.decide(str(tmp_path))
    assert d['resolution'] == 112 and d['coarse_size_flag'] == 112
    probe(tmp_path, 'fc224', 150.0, batch=64)                                   # only fits a smaller batch
    assert pt.decide(str(tmp_path))['resolution'] == 112
    probe(tmp_path, 'lowmem', 200.0)
    probe(tmp_path, 'fc224', 300.0)
    d = pt.decide(str(tmp_path))                                                # 200 (2*200 + 8*300) / 3600 = 155.6 h
    assert d['drop_A2'] and 'R3' not in d['runs'] and 'R6' not in d['runs'] and 'R4' in d['runs']
    assert abs(d['gpu_hours_planned'] - 200 * (2 * 200 + 6 * 300) / 3600) < 1e-9


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
    assert 'old' not in text and '**224 px**' in text and text.startswith('head') and text.endswith('tail\n')


def r0_repro(all_best=0.905, **over):
    rec = {'run_id': 'R0-repro', 'base_method': 'Hyp-SelEx', 'dataset': 'cub', 'c': 0.1, 'r': 1.5,
           'fix_label_order': False, 'split_train_val': False, 'freqcone': {'enable': False},
           'upstream_best': {'epoch': 150, 'all': all_best, 'old': 0.85, 'new': 0.93},
           'test_last': {'all': 0.89, 'old': 0.85, 'new': 0.91}}
    rec.update(over)
    return rec


def test_g0():
    assert pt.g0(r0_repro(0.905))['pass'] and pt.g0(r0_repro(0.887))['pass']
    assert not pt.g0(r0_repro(0.886))['pass'] and not pt.g0(r0_repro(0.93))['pass']
    assert not pt.g0(r0_repro(0.905, fix_label_order=True))['pass']            # must be the upstream configuration
    assert not pt.g0(r0_repro(0.905, c=0.05, r=2.3))['pass']
    assert not pt.g0(r0_repro(0.905, upstream_best={'epoch': 1, 'all': None}))['pass']


def test_budget_overrun(tmp_path):
    decision = {'planned_hours': {'R0-repro': 10.0, 'R0': 10.0, 'R1': 15.0}}
    for rid, hours in (('R0-repro', 12.0), ('R0', 12.5)):
        (tmp_path / f'{rid}.json').write_text(json.dumps({'cost': {'wall_s': hours * 3600}}))
    assert not pt.budget(str(tmp_path), decision)['over']                       # 24.5 h for 20 h: within 30 %
    (tmp_path / 'R0.json').write_text(json.dumps({'cost': {'wall_s': 15 * 3600}}))
    assert pt.budget(str(tmp_path), decision)['over']                           # 27 h > 26 h


def git(repo, *args):
    subprocess.run(['git', *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / 'repo'
    root.mkdir()
    git(root, 'init', '-q')
    git(root, 'config', 'user.email', 'test@example.com')
    git(root, 'config', 'user.name', 'test')
    (root / 'config.py').write_text('cub_root = "{DATA_PATH}"\n')
    (root / 'cutoffs.json').write_text('{}')
    os.makedirs(root / 'results_ssb' / 'profile_cub')
    (root / 'results_ssb' / 'profile_cub' / 'decision.json').write_text('{}')
    git(root, 'add', '-A')
    git(root, 'commit', '-q', '-m', 'init')
    git(root, 'tag', 'pilot-v1')
    return root


def test_preflight(repo, monkeypatch):
    monkeypatch.setattr(pt, 'REPO_ROOT', str(repo))
    monkeypatch.chdir(repo)
    results, decision = 'results_ssb/pilot_cub', 'results_ssb/profile_cub/decision.json'
    (repo / 'config.py').write_text('cub_root = "/data/cub"\n')                # filled-in paths are allowed
    assert pt.preflight(results, decision, 'results_ssb/smoke_cub') == []
    os.makedirs(repo / 'results_ssb' / 'smoke_cub')
    assert any('smoke' in p for p in pt.preflight(results, decision, 'results_ssb/smoke_cub'))
    os.rmdir(repo / 'results_ssb' / 'smoke_cub')
    (repo / 'cutoffs.json').write_text('{"cub": 1}')
    assert any('cutoffs.json' in p for p in pt.preflight(results, decision, 'results_ssb/smoke_cub'))
    git(repo, 'checkout', '-q', '--', 'cutoffs.json')
    os.makedirs(repo / results)
    (repo / results / 'selection.json').write_text('{}')
    git(repo, 'add', f'{results}/selection.json')
    git(repo, 'commit', '-q', '-m', 'lock')                                     # restart after locking: allowed
    assert pt.preflight(results, decision, 'results_ssb/smoke_cub') == []
    (repo / 'other.py').write_text('x = 1\n')
    git(repo, 'add', 'other.py')
    git(repo, 'commit', '-q', '-m', 'code change')
    assert any('differs from tag' in p for p in pt.preflight(results, decision, 'results_ssb/smoke_cub'))


def run_record(rid, all_, new, fc=True, w=0.03, ap='A1', variant='v2'):
    return {'schema': SCHEMA, 'run_id': rid, 'git_commit': 'abc', 'attempt': 1, 'fix_label_order': True,
            'split_train_val': True, 'c': 0.1,
            'freqcone': {'enable': fc, 'variant': variant if fc else None, 'w_ent': w, 'aperture': ap},
            'val': {'value': 0.8}, 'test_last': {'all': all_, 'old': 0.85, 'new': new},
            'geometry_final': {'depth_full': 3.0, 'angle_b_u_deg': 30.0, 'clip_frac': 0.97,
                               'per_aperture': {'A1': {'depth_b': 1.282, 'frac_deeper_than_b': 1.0, 'in_cone_b_u': 0.4}}},
            'cost': {'time_per_epoch_s': 100.0, 'peak_mem_gb': 12.0}}


def test_report_refuses_until_the_selection_is_committed(repo):
    results = repo / 'results_ssb' / 'pilot_cub'
    os.makedirs(results)
    (results / 'R0.json').write_text(json.dumps(run_record('R0', 0.80, 0.78, fc=False)))
    (results / 'R2.json').write_text(json.dumps(run_record('R2', 0.80, 0.795)))
    sel = {'selected': {'run_id': 'R2', 'w_ent': 0.03, 'aperture': 'A1'}, 'm_star': 'R2', 'r8_config': None,
           'table': [{'run_id': 'R2', 'SE': 0.0}],
           'inputs_sha256': {'R2.json': __import__('hashlib').sha256((results / 'R2.json').read_bytes()).hexdigest()}}
    (results / 'selection.json').write_text(json.dumps(sel))
    with pytest.raises(rp.LockError, match='not committed'):
        rp.report(str(results), repo=str(repo))
    git(repo, 'add', '-f', 'results_ssb/pilot_cub/selection.json')
    git(repo, 'commit', '-q', '-m', 'lock')
    out, md = rp.report(str(results), repo=str(repo))
    assert abs(out['g1']['delta_new'] - 1.5) < 1e-6 and abs(out['g1']['delta_all']) < 1e-6 and out['g1']['pass']
    assert 'R8' in out['not_run'] and 'G1' in md and 'PASS' in md
    (results / 'R2.json').write_text(json.dumps(run_record('R2', 0.90, 0.90)))   # tampering after the selection
    with pytest.raises(rp.LockError, match='changed after the selection'):
        rp.report(str(results), repo=str(repo))
    (results / 'R2.json').write_text(json.dumps(run_record('R2', 0.80, 0.795)))
    (results / 'selection.json').write_text(json.dumps({**sel, 'm_star': 'R0'}))
    with pytest.raises(rp.LockError, match='changed after its commit'):
        rp.report(str(results), repo=str(repo))


def test_g1_thresholds():
    runs = {'R0': run_record('R0', 0.800, 0.780, fc=False), 'R5': run_record('R5', 0.796, 0.790)}
    g = rp.g1(runs, 'R5')
    assert g['pass'] and abs(g['delta_new'] - 1.0) < 1e-6 and abs(g['delta_all'] + 0.4) < 1e-6
    runs['R5'] = run_record('R5', 0.794, 0.795)                                 # All drops by 0.6
    assert not rp.g1(runs, 'R5')['pass']
    runs['R5'] = run_record('R5', 0.81, 0.789)                                  # New +0.9 only
    assert not rp.g1(runs, 'R5')['pass']
