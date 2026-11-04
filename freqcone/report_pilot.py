"""Pilot report: the only script that prints test numbers of R1-R8, and only after the selection is locked.

Gate     RESULTS_DIR/selection.json must be tracked by git, committed and unchanged since (git diff HEAD is
         empty for it), and every input file it lists must still have the recorded SHA-256.
G1       The selected configuration against R0 on the last-epoch test numbers (v2 protocol): pass when
         New_sel - New_R0 >= +1.0 and All_sel - All_R0 >= -0.5 (points). One seed: a cheap filter, not evidence.
Regime   R0's mean sharp-view depth against depth_b of A1, A2 and A3 at the run's c (deep or shallow regime).

usage: python -m freqcone.report_pilot RESULTS_DIR
Writes RESULTS_DIR/pilot_report.json and pilot_report.md.
"""
import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys

from freqcone.hypcd import REPO_ROOT, SCHEMA

G1_NEW, G1_ALL = 1.0, -0.5
ORDER = ['R0-repro', 'R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']


class LockError(RuntimeError):
    pass


def check_locked(selection_path, results_dir, repo=REPO_ROOT):
    rel = os.path.relpath(os.path.abspath(selection_path), os.path.abspath(repo)).replace('\\', '/')
    if subprocess.run(['git', 'ls-files', '--error-unmatch', rel], cwd=repo, capture_output=True).returncode:
        raise LockError(f'{rel} is not committed: commit the selection before looking at test numbers')
    if subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', rel], cwd=repo).returncode:
        raise LockError(f'{rel} changed after its commit')
    with open(selection_path) as f:
        sel = json.load(f)
    for name, digest in sel['inputs_sha256'].items():
        with open(os.path.join(results_dir, name), 'rb') as f:
            if hashlib.sha256(f.read()).hexdigest() != digest:
                raise LockError(f'{name} changed after the selection')
    return sel


def load_runs(results_dir):
    runs = {}
    for path in glob.glob(os.path.join(results_dir, '*.json')):
        with open(path) as f:
            rec = json.load(f)
        if isinstance(rec, dict) and rec.get('schema') == SCHEMA and rec.get('test_last'):
            runs[rec['run_id']] = rec
    return runs


def row(rec, sel_table):
    fc = rec['freqcone']
    geo = rec.get('geometry_final') or {}
    ap = (geo.get('per_aperture') or {}).get(fc['aperture'], {})
    s = next((r for r in sel_table if r['run_id'] == rec['run_id']), None)
    return {'run_id': rec['run_id'], 'variant': fc['variant'] or 'baseline',
            'w_ent': fc['w_ent'] if fc['enable'] else None, 'aperture': fc['aperture'] if fc['enable'] else None,
            'fix_label_order': rec['fix_label_order'], 'split_train_val': rec['split_train_val'],
            'val': (rec.get('val') or {}).get('value'), 'val_SE_vs_best': s['SE'] if s else None,
            'all': rec['test_last']['all'], 'old': rec['test_last']['old'], 'new': rec['test_last']['new'],
            'depth_full': geo.get('depth_full'), 'depth_b': ap.get('depth_b'), 'in_cone_b_u': ap.get('in_cone_b_u'),
            'angle_b_u_deg': geo.get('angle_b_u_deg'), 'clip_frac': geo.get('clip_frac'),
            'time_per_epoch_s': rec['cost']['time_per_epoch_s'], 'peak_mem_gb': rec['cost']['peak_mem_gb'],
            'git_commit': rec['git_commit'], 'attempt': rec.get('attempt')}


def g1(runs, selected_id):
    if 'R0' not in runs or selected_id not in runs:
        return None
    sel, base = runs[selected_id]['test_last'], runs['R0']['test_last']
    d_new, d_all = 100 * (sel['new'] - base['new']), 100 * (sel['all'] - base['all'])
    return {'selected': selected_id, 'delta_new': d_new, 'delta_all': d_all, 'threshold_new': G1_NEW,
            'threshold_all': G1_ALL, 'pass': bool(d_new >= G1_NEW and d_all >= G1_ALL)}


def regime(runs):
    if 'R0' not in runs:
        return None
    geo = runs['R0'].get('geometry_final') or {}
    return {'R0_depth_full': geo.get('depth_full'), 'c': runs['R0']['c'],
            'per_aperture': {k: {'depth_b': v['depth_b'], 'frac_deeper_than_b': v['frac_deeper_than_b'],
                                 'in_cone_b_u': v['in_cone_b_u']} for k, v in (geo.get('per_aperture') or {}).items()}}


def fmt(x, scale=1.0, nd=1):
    return '-' if x is None else f'{scale * x:.{nd}f}'


def markdown(rows, g, reg, sel):
    lines = ['| run | variant | λ_ent | aperture | val | All | Old | New | depth(u) | δ(b) | in cone | ∠(b,u) | s/epoch | GB |',
             '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in rows:
        lines.append(f"| {r['run_id']} | {r['variant']} | {fmt(r['w_ent'], nd=2)} | {r['aperture'] or '-'} | "
                     f"{fmt(r['val'], 100)} | {fmt(r['all'], 100)} | {fmt(r['old'], 100)} | {fmt(r['new'], 100)} | "
                     f"{fmt(r['depth_full'], nd=3)} | {fmt(r['depth_b'], nd=3)} | {fmt(r['in_cone_b_u'], 100)} | "
                     f"{fmt(r['angle_b_u_deg'])} | {fmt(r['time_per_epoch_s'])} | {fmt(r['peak_mem_gb'])} |")
    lines.append('')
    lines.append(f"Selected (validation only): {sel['selected']['run_id']} (λ_ent {sel['selected']['w_ent']}, "
                 f"{sel['selected']['aperture']}); m* = {sel['m_star']}.")
    if g:
        lines.append(f"G1: ΔNew {g['delta_new']:+.2f} (≥ {g['threshold_new']:+.1f}), ΔAll {g['delta_all']:+.2f} "
                     f"(≥ {g['threshold_all']:+.1f}) → {'PASS' if g['pass'] else 'FAIL'}.")
    if reg:
        parts = [f"{k}: δ(b) {v['depth_b']:.3f}, deeper {100 * v['frac_deeper_than_b']:.0f}%" for k, v in reg['per_aperture'].items()]
        lines.append(f"Regime (R0, c = {reg['c']}): mean depth(u) {fmt(reg['R0_depth_full'], nd=3)}; " + '; '.join(parts) + '.')
    return '\n'.join(lines) + '\n'


def report(results_dir, repo=REPO_ROOT):
    sel = check_locked(os.path.join(results_dir, 'selection.json'), results_dir, repo)
    runs = load_runs(results_dir)
    rows = [row(runs[k], sel['table']) for k in ORDER if k in runs]
    out = {'selection': {'selected': sel['selected'], 'm_star': sel['m_star'], 'r8_config': sel['r8_config']},
           'rows': rows, 'g1': g1(runs, sel['selected']['run_id']), 'regime': regime(runs),
           'not_run': [k for k in ORDER if k not in runs]}
    with open(os.path.join(results_dir, 'pilot_report.json'), 'w') as f:
        json.dump(out, f, indent=1)
    md = markdown(rows, out['g1'], out['regime'], sel)
    with open(os.path.join(results_dir, 'pilot_report.md'), 'w', encoding='utf-8') as f:
        f.write(md)
    return out, md


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('results_dir')
    a = ap.parse_args(argv)
    try:
        out, md = report(a.results_dir)
    except LockError as e:
        print(f'refused: {e}', file=sys.stderr)
        return 2
    print(md)
    if out['not_run']:
        print(f"not run: {', '.join(out['not_run'])}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
