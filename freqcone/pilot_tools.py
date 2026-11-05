"""Gates and checks of the Stanford Cars plan (rules fixed before any SSB run).

  python -m freqcone.pilot_tools check RUN_DIR...            run.json + JSONL files (plan 2.1); --profile for probes
  python -m freqcone.pilot_tools decide PROFILE_DIR           coarse-view resolution and budget level A / B (3.1-3.2)
  python -m freqcone.pilot_tools g0 RUN_DIR                   reproduction gate G0 (3.3)
  python -m freqcone.pilot_tools budget RUNS_ROOT DECISION    stop when runs take > 30 % longer than planned
  python -m freqcone.pilot_tools preflight RUNS_ROOT DECISION tagged code, locked inputs, no smoke results
Exit status 0 = pass; anything else = stop and report.
"""
import argparse
import json
import os
import re
import subprocess
import sys

from freqcone.hypcd import REPO_ROOT, SCHEMA
from freqcone.runs import read_jsonl, read_run_json

# ---------------------------------------------------------------------------------------------- 2.1 schema
REQUIRED_TOP = ['schema', 'run_id', 'git_commit', 'command', 'args', 'dataset', 'seed', 'c', 'r', 'dist_impl',
                'gpu_name', 'torch_version', 'cuda_version', 'dinov2_checkpoint', 'data_lists', 'freqcone', 'files',
                'diagnostics', 'cost', 'fix_label_order', 'split_train_val', 'eval_every_epoch']
REQUIRED_FC = ['enable', 'variant', 'w_dir', 'w_ent', 'w_con', 'K', 'r_a', 'r_b', 'psi_a', 'psi_b', 'depth_a',
               'depth_b', 'f_a', 'f_b', 'coarse_res']
REQUIRED_COST = ['time_per_epoch_s', 'peak_mem_gb']
REQUIRED_VAL_LINE = ['epoch', 'val_acc', 'losses', 'sup_con', 'fc', 'lr', 'time_s']
REQUIRED_DATA_COUNTS = ['n_labelled', 'n_unlabelled', 'n_val', 'n_test_roi', 'labelled_per_known_min',
                        'labelled_per_known_max']
REQUIRED_DIAG = ['depth_full', 'in_cone', 'clip_frac', 'angle_b_u']


def _has_test_numbers(obj):
    if isinstance(obj, dict):
        if {'all', 'old', 'new'} <= set(obj) or 'test_last' in obj or 'upstream_best' in obj:
            return True
        return any(_has_test_numbers(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_has_test_numbers(v) for v in obj)
    return False


def check_run_dir(run_dir, profile=False):
    """List of problems (empty = valid)."""
    try:
        rec = read_run_json(run_dir)
    except (OSError, ValueError) as e:
        return [str(e)]
    errs = [f'missing {k}' for k in REQUIRED_TOP if k not in rec]
    errs += [f'missing freqcone.{k}' for k in REQUIRED_FC if k not in (rec.get('freqcone') or {})]
    errs += [f'missing cost.{k}' for k in REQUIRED_COST if k not in (rec.get('cost') or {})]
    if _has_test_numbers(rec):
        errs.append('run.json holds test numbers (they belong in epochs_test.jsonl only)')
    if profile:
        if (rec.get('profile') or {}).get('est_epoch_s') is None:
            errs.append('missing profile.est_epoch_s')
        return errs
    if 'geometry_final' not in rec:
        errs.append('missing geometry_final')
    errs += [f'missing data_counts.{k}' for k in REQUIRED_DATA_COUNTS if k not in (rec.get('data_counts') or {})]
    hist = rec.get('history') or []
    if not hist or not all((h.get('sup_con') or {}).get('n_labelled') for h in hist):
        errs.append('history lacks the supervised SupCon record of some epoch, or it saw no labelled image')
    if rec.get('split_train_val'):
        val = rec.get('val') or {}
        errs += [f'missing val.{k}' for k in ('metric', 'value', 'per_sample_true', 'per_sample_pred', 'val_list_sha256')
                 if k not in val]
    paths = {k: os.path.join(run_dir, f) for k, f in (('val', 'epochs_val.jsonl'), ('test', 'epochs_test.jsonl'))}
    if not os.path.exists(paths['test']):
        return errs + ['missing epochs_test.jsonl']
    tests = read_jsonl(paths['test'])
    want = list(range(1, int(rec['epochs']) + 1))
    if [t['epoch'] for t in tests] != want:
        errs.append(f'epochs_test.jsonl does not cover epochs 1..{rec["epochs"]} exactly once')
    triple = lambda x: isinstance(x, dict) and {'all', 'old', 'new'} <= set(x)
    if not all(triple(t.get('gcd')) and triple(t.get('disjoint_test')) for t in tests):
        errs.append('epochs_test.jsonl lines lack gcd{all,old,new} / disjoint_test{all,old,new}')
    if tests and not triple(tests[-1].get('eval')):
        errs.append('the last epochs_test.jsonl line lacks the last-epoch evaluation (eval)')
    if rec.get('eval_every_epoch'):
        vals = read_jsonl(paths['val']) if os.path.exists(paths['val']) else []
        if [v['epoch'] for v in vals] != want or not all(triple(t.get('eval')) for t in tests):
            errs.append(f'per-epoch evaluation does not cover epochs 1..{rec["epochs"]} exactly once')
        errs += [f'epochs_val.jsonl missing {k}' for k in REQUIRED_VAL_LINE if vals and k not in vals[-1]]
        if rec.get('split_train_val') and vals and 'val_per_sample_pred' not in vals[-1]:
            errs.append('last epochs_val.jsonl line lacks the per-image predictions')
    diag = rec.get('diagnostics') or []
    if not diag:
        errs.append('no step diagnostics recorded')
    elif any(k not in diag[0] for k in REQUIRED_DIAG):
        errs.append('step diagnostics lack depth_full / in_cone / clip_frac / angle_b_u')
    elif (rec.get('freqcone') or {}).get('enable') and (diag[0].get('in_cone') is None or diag[0].get('angle_b_u') is None):
        errs.append('FreqCone run without in_cone / angle_b_u diagnostics')
    return errs


# ---------------------------------------------------------------------------------------------- 3.1-3.2 decision
RATIO_MAX, LEVEL_A_MAX_RUN_H, EPOCHS = 1.8, 10.0, 200
LEVEL_RUNS = {   # (run id, arm); R0-repro is conditional on G0 and listed apart
    'A': [('R0', 'base'), ('R1', 'fc'), ('R2', 'fc'), ('R3', 'fc'), ('base_s8', 'base'), ('base_s9', 'base'),
          ('base_s10', 'base'), ('fc_s8', 'fc'), ('fc_s9', 'fc'), ('fc_s10', 'fc')],
    'B': [('base_s8', 'base'), ('base_s9', 'base'), ('base_s10', 'base'), ('fc_s8', 'fc'), ('fc_s9', 'fc'),
          ('fc_s10', 'fc')],
}
README_START, README_END = '<!-- pilot-decision:start -->', '<!-- pilot-decision:end -->'


def _probe(profile_dir, tag):
    path = os.path.join(profile_dir, tag)
    try:
        rec = read_run_json(path)
    except (OSError, ValueError):
        return None
    return rec if (rec.get('profile') or {}).get('est_epoch_s') is not None else None


def decide(profile_dir, epochs=EPOCHS, upstream_batch=128):
    base, fc224, fc112 = (_probe(profile_dir, t) for t in ('lowmem', 'fc224', 'fc112'))
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
    hours = {'base': epochs * t_base / 3600, 'fc': epochs * t_fc / 3600}
    level = 'A' if max(hours.values()) <= LEVEL_A_MAX_RUN_H else 'B'
    totals = {lv: sum(hours[arm] for _, arm in runs) for lv, runs in LEVEL_RUNS.items()}
    return {
        'rule': (f'resolution: 224 px if the 224 px probe ran at batch {upstream_batch} and its epoch time is at most '
                 f'{RATIO_MAX} x the baseline, else 112 px for every FreqCone run; level A if one run takes at most '
                 f'{LEVEL_A_MAX_RUN_H} h ({epochs} epochs), else level B; GPU-hours = sum over the planned runs of '
                 f'{epochs} x t_epoch'),
        'resolution': res, 'coarse_res_flag': 0 if res == 224 else 112,
        't_epoch_s': {'baseline_lowmem': t_base, 'fc224': fc224['profile']['est_epoch_s'] if fc224 else None,
                      'fc112': fc112['profile']['est_epoch_s'] if fc112 else None},
        'ratio_224': fc224['profile']['est_epoch_s'] / t_base if fc224 else None,
        'peak_mem_gb': {t: (r or {}).get('cost', {}).get('peak_mem_gb') for t, r in
                        (('lowmem', base), ('fc224', fc224), ('fc112', fc112))},
        'hours_per_run': hours, 'level': level, 'gpu_hours': totals,
        'r0_repro_hours_if_needed': hours['base'],
        'planned_hours': {**{rid: hours[arm] for rid, arm in LEVEL_RUNS[level]}, 'R0-repro': hours['base']},
    }


def readme_block(d):
    t = d['t_epoch_s']
    fmt = lambda x: '–' if x is None else f'{x:.0f} s'
    return '\n'.join([
        README_START,
        f"Quyết định từ các probe profile (luật trong `freqcone/pilot_tools.py`): view thô **{d['resolution']} px**; "
        f"thời gian một epoch ước tính: baseline {fmt(t['baseline_lowmem'])}, FreqCone 224 px {fmt(t['fc224'])}, "
        f"112 px {fmt(t['fc112'])}. Một run 200 epoch: baseline {d['hours_per_run']['base']:.1f} h, FreqCone "
        f"{d['hours_per_run']['fc']:.1f} h. **Mức {d['level']}** (mức A: {d['gpu_hours']['A']:.1f} GPU-giờ; "
        f"mức B: {d['gpu_hours']['B']:.1f} GPU-giờ; R0-repro nếu cần: +{d['r0_repro_hours_if_needed']:.1f} h).",
        README_END])


def write_readme(d, readme=os.path.join(REPO_ROOT, 'README.md')):
    with open(readme, encoding='utf-8') as f:
        text = f.read()
    pattern = re.compile(re.escape(README_START) + '.*?' + re.escape(README_END), re.S)
    if not pattern.search(text):
        raise RuntimeError(f'{readme} has no pilot-decision markers')
    with open(readme, 'w', encoding='utf-8') as f:
        f.write(pattern.sub(lambda _: readme_block(d), text))


# ---------------------------------------------------------------------------------------------- 3.3 G0
PUBLISHED = {   # HypCD (CVPR 2025) Table 1, DINOv2, one published run each: All / Old / New in %
    ('Hyp-SelEx', 'scars'): {'all': 83.8, 'old': 93.3, 'new': 79.2},
    ('Hyp-SelEx', 'cub'): {'all': 90.7, 'old': 85.3, 'new': 93.4},
}
G0_TOL = 2.0
UPSTREAM_CR = {'cub': (0.1, 1.5), 'scars': (0.1, 2.0), 'aircraft': (0.05, 1.5)}   # scripts/train_HypSelEx.cmd, DINOv2
UPSTREAM_SEED = 1                                                                  # train_HypSelEx.py default


def g0(run_dir):
    from freqcone.summarize_run import table
    rec = read_run_json(run_dir)
    problems = []
    if (rec.get('freqcone') or {}).get('enable'):
        problems.append('G0 is checked on a baseline run')
    if (rec.get('c'), rec.get('r')) != UPSTREAM_CR.get(rec.get('dataset')):
        problems.append(f"c, r = {rec.get('c')}, {rec.get('r')}; upstream uses {UPSTREAM_CR.get(rec.get('dataset'))}")
    if rec.get('run_id') == 'R0-repro' and (rec.get('fix_label_order') or rec.get('split_train_val')
                                            or rec.get('seed') != UPSTREAM_SEED):
        problems.append('R0-repro must be upstream as is: label-order bug kept, no validation split, seed 1')
    rows = {r['key']: r for r in table(read_jsonl(os.path.join(run_dir, 'epochs_val.jsonl')),
                                       read_jsonl(os.path.join(run_dir, 'epochs_test.jsonl')))}
    up, last = rows['gcd'], rows['last']
    pub = PUBLISHED.get((rec.get('base_method'), rec.get('dataset')))
    out = {'run_id': rec.get('run_id'), 'base_method': rec.get('base_method'), 'dataset': rec.get('dataset'),
           'published': pub, 'gcd_row': up, 'last_row': last, 'tolerance': G0_TOL, 'problems': problems}
    if pub is None or up['all'] is None:
        problems.append('no published reference, or no per-epoch gcd / disjoint_test records')
        out.update(delta_all=None, passed=False)
    else:
        delta = 100 * up['all'] - pub['all']
        out.update(delta_all=delta, passed=bool(abs(delta) <= G0_TOL and not problems))
    return out


# ---------------------------------------------------------------------------------------------- budget, preflight
BUDGET_SLACK = 1.3


def budget(runs_root, decision):
    actual = planned = 0.0
    done = []
    for rid, hours in decision['planned_hours'].items():
        path = os.path.join(runs_root, rid, 'run.json')
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


def check_environment():
    """Problems that would otherwise surface only after the first training epoch."""
    import inspect

    from kmeans_pytorch import kmeans
    if 'tqdm_flag' in inspect.signature(kmeans).parameters:
        return []
    return ["kmeans_pytorch has no tqdm_flag, which upstream's test_kmeans passes: the PyPI build is too old. Install "
            "it from https://github.com/subhadarship/kmeans_pytorch (docs/UPSTREAM_README.md, README section 2)"]


def preflight(runs_root, decision_path, smoke_dir):
    """Code = the latest pilot-v* tag (commits after it may only touch results under RUNS_ROOT); config.py may
    hold local paths; cutoffs.json and the decision committed and unchanged; no smoke results."""
    problems = []
    runs_rel = os.path.relpath(os.path.abspath(runs_root), REPO_ROOT).replace('\\', '/').rstrip('/') + '/'
    code, tag = _git('describe', '--tags', '--abbrev=0', '--match', 'pilot-v*')
    if code or not tag:
        problems.append('no pilot-v* tag is reachable from HEAD')
    else:
        _, changed = _git('diff', '--name-only', tag, 'HEAD')
        extra = [c for c in changed.split('\n') if c and not c.startswith(runs_rel)]
        if extra:
            problems.append(f'HEAD differs from tag {tag} outside {runs_rel}: {extra}')
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
    p.add_argument('run_dirs', nargs='+')
    p.add_argument('--profile', action='store_true')
    p = sub.add_parser('decide')
    p.add_argument('profile_dir')
    p.add_argument('--no-readme', action='store_true')
    p = sub.add_parser('g0')
    p.add_argument('run_dir')
    p = sub.add_parser('budget')
    p.add_argument('runs_root')
    p.add_argument('decision')
    sub.add_parser('env')
    p = sub.add_parser('preflight')
    p.add_argument('runs_root')
    p.add_argument('decision')
    p.add_argument('--smoke_dir', default='runs/smoke_scars')
    a = ap.parse_args(argv)
    from freqcone.runs import utf8_stdout
    utf8_stdout()

    if a.cmd == 'check':
        bad = 0
        for d in a.run_dirs:
            errs = check_run_dir(d, profile=a.profile)
            print(f"{'OK  ' if not errs else 'FAIL'} {d}" + ('' if not errs else ': ' + '; '.join(errs)))
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
        out = g0(a.run_dir)
        with open(os.path.join(a.run_dir, 'g0.json'), 'w') as f:
            json.dump(out, f, indent=1)
        up, last, pub = out['gcd_row'], out['last_row'], out['published'] or {}
        pct = lambda x: '-' if x is None else f'{100 * x:.2f}'
        print(f"G0 {out['base_method']} / {out['dataset']} ({out['run_id']}): GCD standard row (epoch {up['epoch']}) All "
              f"{pct(up['all'])} / Old {pct(up['old'])} / New {pct(up['new'])} vs published {pub.get('all')} / "
              f"{pub.get('old')} / {pub.get('new')}; delta All "
              f"{'-' if out['delta_all'] is None else round(out['delta_all'], 2)} (tolerance ±{G0_TOL}); last epoch "
              f"All {pct(last['all'])}; {'PASS' if out['passed'] else 'FAIL'}"
              + ('' if not out['problems'] else f"; problems: {out['problems']}"))
        return 0 if out['passed'] else 1
    if a.cmd == 'budget':
        with open(a.decision) as f:
            out = budget(a.runs_root, json.load(f))
        print(f"budget: {out['actual_hours']:.1f} h used for {out['planned_hours']:.1f} h planned "
              f"({', '.join(out['runs_done'])})" + (' -> more than 30 % over: report before going on' if out['over'] else ''))
        return 3 if out['over'] else 0
    if a.cmd == 'env':
        problems = check_environment()
        print('\n'.join(problems) or 'environment OK')
        return 1 if problems else 0
    if a.cmd == 'preflight':
        problems = preflight(a.runs_root, a.decision, a.smoke_dir)
        for pr in problems:
            print(f'preflight: {pr}')
        return 1 if problems else 0
    return 2


if __name__ == '__main__':
    sys.exit(main())
