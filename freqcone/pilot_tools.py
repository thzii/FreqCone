"""Gates and checks of the CUB pilot (rules fixed on 2026-11-04, before any SSB run).

  python -m freqcone.pilot_tools check FILE...            run-record schema (section 2.7); --profile for probes
  python -m freqcone.pilot_tools decide PROFILE_DIR        coarse-view resolution and budget (section 3.3)
  python -m freqcone.pilot_tools g0 R0_REPRO_JSON          reproduction gate G0 (section 3.4)
  python -m freqcone.pilot_tools budget RESULTS DECISION   stop when runs take > 30 % longer than planned
  python -m freqcone.pilot_tools preflight RESULTS DECISION  tagged commit, locked inputs, no smoke results
Exit status 0 = pass; anything else = stop and report.
"""
import argparse
import json
import os
import re
import subprocess
import sys

from freqcone.hypcd import REPO_ROOT, SCHEMA

# ---------------------------------------------------------------------------------------------- schema
REQUIRED_TOP = ['schema', 'run_id', 'git_commit', 'command', 'args', 'dataset', 'seed', 'c', 'r', 'dist_impl',
                'gpu_name', 'torch_version', 'cuda_version', 'freqcone', 'val', 'test_last', 'upstream_best',
                'diagnostics', 'cost', 'fix_label_order', 'split_train_val', 'geometry_final']
REQUIRED_FC = ['enable', 'variant', 'w_dir', 'w_ent', 'w_con', 'K', 'r_a', 'r_b', 'psi_a', 'psi_b', 'depth_a',
               'depth_b', 'f_a', 'f_b', 'coarse_res']
REQUIRED_VAL = ['metric', 'value', 'per_sample_true', 'per_sample_pred', 'val_list_sha256']
REQUIRED_DIAG = ['depth_full', 'in_cone', 'clip_frac', 'angle_b_u']
REQUIRED_COST = ['time_per_epoch_s', 'peak_mem_gb']


def check_record(rec, profile=False):
    """List of problems (empty = valid)."""
    errs = []
    top = [k for k in REQUIRED_TOP if not (profile and k in ('val', 'test_last', 'upstream_best', 'geometry_final'))]
    errs += [f'missing {k}' for k in top if k not in rec]
    if rec.get('schema') != SCHEMA:
        errs.append(f"schema {rec.get('schema')!r} != {SCHEMA}")
    fc = rec.get('freqcone') or {}
    errs += [f'missing freqcone.{k}' for k in REQUIRED_FC if k not in fc]
    errs += [f'missing cost.{k}' for k in REQUIRED_COST if k not in (rec.get('cost') or {})]
    if profile:
        if (rec.get('profile') or {}).get('est_epoch_s') is None:
            errs.append('missing profile.est_epoch_s')
        return errs
    if rec.get('split_train_val'):
        val = rec.get('val') or {}
        errs += [f'missing val.{k}' for k in REQUIRED_VAL if k not in val]
        if val and len(val.get('per_sample_true', [])) != len(val.get('per_sample_pred', [])):
            errs.append('val per-sample lists differ in length')
    for key, fields in (('test_last', ('all', 'old', 'new')), ('upstream_best', ('epoch', 'all', 'old', 'new'))):
        errs += [f'missing {key}.{k}' for k in fields if k not in (rec.get(key) or {})]
    diag = rec.get('diagnostics') or []
    if not diag:
        errs.append('no diagnostics recorded')
    for d in diag[:1]:
        errs += [f'missing diagnostics.{k}' for k in REQUIRED_DIAG if k not in d]
        if fc.get('enable') and (d.get('in_cone') is None or d.get('angle_b_u') is None):
            errs.append('FreqCone run without in_cone / angle_b_u diagnostics')
        if d.get('depth_full') is None:
            errs.append('no sharp-view depth in the diagnostics')
    return errs


# ---------------------------------------------------------------------------------------------- 3.3 decision
RATIO_MAX, BUDGET_H, EPOCHS = 1.8, 100.0, 200
BASE_RUNS = ['R0-repro', 'R0']
FC_RUNS = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']
A2_RUNS = ['R3', 'R6']        # lambda_ent 0.03 and 0.1 with aperture A2 (section 3.5)
README_START, README_END = '<!-- pilot-decision:start -->', '<!-- pilot-decision:end -->'


def _profile(profile_dir, tag):
    path = os.path.join(profile_dir, f'{tag}.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rec = json.load(f)
    return rec if (rec.get('profile') or {}).get('est_epoch_s') is not None else None


def decide(profile_dir, epochs=EPOCHS, upstream_batch=128):
    base, fc224, fc112 = (_profile(profile_dir, t) for t in ('lowmem', 'fc224', 'fc112'))
    if base is None:
        raise RuntimeError('the lowmem baseline probe did not finish: report before going on')
    t_base = base['profile']['est_epoch_s']
    ok224 = bool(fc224) and fc224['batch_size'] == upstream_batch and \
        fc224['profile']['est_epoch_s'] <= RATIO_MAX * t_base
    if ok224:
        res, t_fc = 224, fc224['profile']['est_epoch_s']
    elif fc112 and fc112['batch_size'] == upstream_batch:
        res, t_fc = 112, fc112['profile']['est_epoch_s']
    else:
        raise RuntimeError('neither FreqCone probe finished at the upstream batch size: report before going on')
    planned = {r: epochs * t_base / 3600 for r in BASE_RUNS}
    planned.update({r: epochs * t_fc / 3600 for r in FC_RUNS})
    total = sum(planned.values())
    drop_a2 = total > BUDGET_H
    runs = [r for r in BASE_RUNS + FC_RUNS if not (drop_a2 and r in A2_RUNS)]
    return {
        'rule': f'224 px if the 224 px probe ran at batch {upstream_batch} and est_epoch_s <= {RATIO_MAX} x baseline, '
                f'else 112 px for every FreqCone run of the pilot and P2; GPU-hours = sum over planned runs of '
                f'{epochs} x t_epoch; above {BUDGET_H} GPU-hours drop aperture A2 (R3, R6)',
        'resolution': res, 'coarse_size_flag': 0 if res == 224 else 112,
        't_epoch_s': {'baseline_lowmem': t_base, 'fc224': fc224['profile']['est_epoch_s'] if fc224 else None,
                      'fc112': fc112['profile']['est_epoch_s'] if fc112 else None},
        'ratio_224': fc224['profile']['est_epoch_s'] / t_base if fc224 else None,
        'peak_mem_gb': {t: (r or {}).get('cost', {}).get('peak_mem_gb') for t, r in
                        (('lowmem', base), ('fc224', fc224), ('fc112', fc112))},
        'gpu_hours_all_runs': total, 'drop_A2': drop_a2, 'runs': runs,
        'planned_hours': {r: planned[r] for r in runs},
        'gpu_hours_planned': sum(planned[r] for r in runs),
    }


def readme_block(d):
    return '\n'.join([
        README_START,
        f"Decided from the profile probes before R1 (rule in `freqcone/pilot_tools.py`): coarse views at "
        f"**{d['resolution']} px**; estimated epoch time baseline {d['t_epoch_s']['baseline_lowmem']:.0f} s, FreqCone "
        f"224 px {d['t_epoch_s']['fc224'] if d['t_epoch_s']['fc224'] is None else round(d['t_epoch_s']['fc224'])} s, "
        f"112 px {d['t_epoch_s']['fc112'] if d['t_epoch_s']['fc112'] is None else round(d['t_epoch_s']['fc112'])} s. "
        f"Planned budget {d['gpu_hours_all_runs']:.1f} GPU-hours for all runs; "
        f"{'aperture A2 dropped (R3, R6)' if d['drop_A2'] else 'all apertures kept'}; runs: {', '.join(d['runs'])}.",
        README_END])


def write_readme(d, readme=os.path.join(REPO_ROOT, 'freqcone', 'README.md')):
    with open(readme, encoding='utf-8') as f:
        text = f.read()
    pattern = re.compile(re.escape(README_START) + '.*?' + re.escape(README_END), re.S)
    if not pattern.search(text):
        raise RuntimeError(f'{readme} has no pilot-decision markers')
    with open(readme, 'w', encoding='utf-8') as f:
        f.write(pattern.sub(lambda _: readme_block(d), text))


# ---------------------------------------------------------------------------------------------- 3.4 G0
PUBLISHED = {   # HypCD Table 1 (CVPR 2025), DINOv2, one published run each: All / Old / New in %
    ('Hyp-SelEx', 'cub'): {'all': 90.7, 'old': 85.3, 'new': 93.4},
    ('Hyp-SimGCD', 'cub'): {'all': 77.6, 'old': 77.9, 'new': 77.4},
}
G0_TOL = 2.0
UPSTREAM_CR = {'cub': (0.1, 1.5), 'scars': (0.1, 2.0), 'aircraft': (0.05, 1.5)}   # scripts/train_HypSelEx.cmd, DINOv2


def g0(rec):
    problems = []
    if rec.get('run_id') != 'R0-repro':
        problems.append(f"run_id {rec.get('run_id')!r}, expected R0-repro")
    if rec.get('fix_label_order') or rec.get('split_train_val') or (rec.get('freqcone') or {}).get('enable'):
        problems.append('R0-repro must keep the upstream configuration (label-order bug kept, no validation split, '
                        'FreqCone off)')
    if (rec.get('c'), rec.get('r')) != UPSTREAM_CR.get(rec.get('dataset')):
        problems.append(f"c, r = {rec.get('c')}, {rec.get('r')}; upstream uses {UPSTREAM_CR.get(rec.get('dataset'))}")
    pub = PUBLISHED.get((rec.get('base_method'), rec.get('dataset')))
    best, last = rec.get('upstream_best') or {}, rec.get('test_last') or {}
    out = {'base_method': rec.get('base_method'), 'dataset': rec.get('dataset'), 'published': pub,
           'upstream_best': best, 'test_last': last, 'tolerance': G0_TOL, 'problems': problems}
    if pub is None or best.get('all') is None:
        out.update(delta_all=None, pass_=False)
        problems.append('no published reference or no upstream_best result')
    else:
        delta = 100 * best['all'] - pub['all']
        out.update(delta_all=delta, pass_=bool(abs(delta) <= G0_TOL and not problems))
    out['pass'] = out.pop('pass_')
    return out


# ---------------------------------------------------------------------------------------------- section 6
BUDGET_SLACK = 1.3


def budget(results_dir, decision):
    actual = planned = 0.0
    done = []
    for rid, hours in decision['planned_hours'].items():
        path = os.path.join(results_dir, f'{rid}.json')
        if os.path.exists(path):
            with open(path) as f:
                rec = json.load(f)
            actual += rec['cost']['wall_s'] / 3600
            planned += hours
            done.append(rid)
    return {'runs_done': done, 'actual_hours': actual, 'planned_hours': planned,
            'over': bool(planned and actual > BUDGET_SLACK * planned)}


def _git(*cmd):
    res = subprocess.run(['git', *cmd], cwd=REPO_ROOT, capture_output=True, text=True)
    return res.returncode, res.stdout.strip()


def preflight(results_dir, decision_path, smoke_dir):
    """HEAD = the latest pilot-v* tag, or that tag plus only the committed selection (restart after locking)."""
    problems = []
    code, tag = _git('describe', '--tags', '--abbrev=0', '--match', 'pilot-v*')
    if code or not tag:
        problems.append('no pilot-v* tag is reachable from HEAD')
    else:
        selection = os.path.relpath(os.path.abspath(os.path.join(results_dir, 'selection.json')),
                                    REPO_ROOT).replace('\\', '/')
        _, changed = _git('diff', '--name-only', tag, 'HEAD')
        extra = [c for c in changed.split('\n') if c and c != selection]
        if extra:
            problems.append(f'HEAD differs from tag {tag} in {extra}; only the committed selection may differ')
    _, modified = _git('diff', '--name-only', 'HEAD')
    extra = [m for m in modified.split('\n') if m and m != 'config.py']
    if extra:
        problems.append(f'tracked files modified besides config.py: {extra}')
    for path in ('cutoffs.json', decision_path):
        rel = os.path.relpath(os.path.abspath(path), REPO_ROOT).replace('\\', '/')
        if _git('ls-files', '--error-unmatch', rel)[0] or _git('diff', '--quiet', 'HEAD', '--', rel)[0]:
            problems.append(f'{rel} is not committed or changed since its commit')
    if os.path.isdir(smoke_dir):
        problems.append(f'smoke results still present: delete {smoke_dir} first')
    return problems


# ---------------------------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('check')
    p.add_argument('files', nargs='+')
    p.add_argument('--profile', action='store_true')
    p = sub.add_parser('decide')
    p.add_argument('profile_dir')
    p.add_argument('--no-readme', action='store_true')
    p = sub.add_parser('g0')
    p.add_argument('path')
    p = sub.add_parser('budget')
    p.add_argument('results_dir')
    p.add_argument('decision')
    p = sub.add_parser('preflight')
    p.add_argument('results_dir')
    p.add_argument('decision')
    p.add_argument('--smoke_dir', default='results_ssb/smoke_cub')
    a = ap.parse_args(argv)

    if a.cmd == 'check':
        bad = 0
        for path in a.files:
            with open(path) as f:
                rec = json.load(f)
            if not (isinstance(rec, dict) and rec.get('schema')):
                continue                       # selection / decision files
            errs = check_record(rec, profile=a.profile)
            print(f"{'OK  ' if not errs else 'FAIL'} {path}" + ('' if not errs else ': ' + '; '.join(errs)))
            bad += bool(errs)
        return 1 if bad else 0
    if a.cmd == 'decide':
        d = decide(a.profile_dir)
        with open(os.path.join(a.profile_dir, 'decision.json'), 'w') as f:
            json.dump(d, f, indent=1)
        if not a.no_readme:
            write_readme(d)
        print(readme_block(d))
        return 0
    if a.cmd == 'g0':
        with open(a.path) as f:
            out = g0(json.load(f))
        with open(os.path.join(os.path.dirname(a.path), 'g0.json'), 'w') as f:
            json.dump(out, f, indent=1)
        b, t, pub = out['upstream_best'], out['test_last'], out['published'] or {}
        print(f"G0 {out['base_method']} / {out['dataset']}: upstream_best (epoch {b.get('epoch')}) All "
              f"{100 * (b.get('all') or 0):.2f} vs published {pub.get('all')} -> delta "
              f"{out['delta_all'] if out['delta_all'] is None else round(out['delta_all'], 2)} (tolerance "
              f"±{G0_TOL}); last epoch All/Old/New {100 * (t.get('all') or 0):.2f} / {100 * (t.get('old') or 0):.2f} / "
              f"{100 * (t.get('new') or 0):.2f}; {'PASS' if out['pass'] else 'FAIL'}"
              + ('' if not out['problems'] else f"; problems: {out['problems']}"))
        return 0 if out['pass'] else 1
    if a.cmd == 'budget':
        with open(a.decision) as f:
            out = budget(a.results_dir, json.load(f))
        print(f"budget: {out['actual_hours']:.1f} h used for {out['planned_hours']:.1f} h planned "
              f"({', '.join(out['runs_done'])})" + (' -> more than 30 % over: report before going on' if out['over'] else ''))
        return 3 if out['over'] else 0
    if a.cmd == 'preflight':
        problems = preflight(a.results_dir, a.decision, a.smoke_dir)
        for pr in problems:
            print(f'preflight: {pr}')
        return 1 if problems else 0
    return 2


if __name__ == '__main__':
    sys.exit(main())
