"""The running order of the plan (docs/PLAN.md, section 3), on the real scripts.

pilot_cars.sh and smoke_gpu.sh run unchanged under bash, with stand-ins for `python` and `git` first on PATH: `python`
logs every call, writes run.json for a training command and answers the gates as the test asks. So the order of the
runs, the flags of each run and the branches of G0 and G1 are checked as the script really executes them.
"""
import json
import os
import re
import shutil
import subprocess
import sys

import pytest

from conftest import ROOT

BASH = shutil.which('bash')
if BASH and 'system32' in BASH.lower():                     # the WSL launcher, which does not see Windows paths
    BASH = shutil.which('bash', path=r'C:\Program Files\Git\bin')
needs_bash = pytest.mark.skipif(BASH is None, reason='needs bash')

PYTHON_STAND_IN = r"""#!/usr/bin/env bash
echo "$*" >> "$CALLS"
case "$1 $2" in
  -c\ *) "$REAL_PYTHON" "$@" | tr -d '\r'; exit "${PIPESTATUS[0]}" ;;
  "-m train.train_HypSelEx")
    args=("$@")
    for ((i = 0; i < ${#args[@]}; i++)); do
      if [ "${args[i]}" = --run_dir ]; then echo '{}' > "${args[i + 1]}/run.json"; fi
    done ;;
  "-m freqcone.select_pilot")
    if [ -z "${SMOKE:-}" ]; then
      echo '{"selected": {"run_id": "R2", "w_ent": 0.1, "aperture": "A1"}}' > runs/selection.json
    fi ;;
  "-m freqcone.pilot_tools")
    if [ "$3" = g0 ]; then for id in $G0_FAIL; do [ "$4" = "runs/$id" ] && exit 1; done; fi ;;
  "-m freqcone.compare_runs")
    case "$*" in *"--gate g1"*) exit "$G1_EXIT" ;; esac ;;
esac
exit 0
"""


@pytest.fixture
def workspace(tmp_path):
    """The scripts in a tree of their own (line endings as git stores them), with cutoffs.json."""
    folder = tmp_path / 'scripts' / 'freqcone'
    folder.mkdir(parents=True)
    for name in ('pilot_cars.sh', 'smoke_gpu.sh', 'hypselex_args.sh'):
        with open(os.path.join(ROOT, 'scripts', 'freqcone', name), encoding='utf-8', newline='') as f:
            (folder / name).write_text(f.read().replace('\r\n', '\n'), encoding='utf-8', newline='\n')
    (tmp_path / 'cutoffs.json').write_text(json.dumps({'scars': {'f_a': 3, 'f_b': 11}}))
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    for name, body in (('python', PYTHON_STAND_IN), ('git', '#!/usr/bin/env bash\nexit 0\n')):
        (bin_dir / name).write_text(body, newline='\n')
        (bin_dir / name).chmod(0o755)
    (tmp_path / 'runs' / 'profile_scars').mkdir(parents=True)
    return tmp_path


def execute(workspace, script, g0_fail='', g1_exit=0, **env):
    calls = workspace / 'calls.log'
    full = dict(os.environ, PATH=str(workspace / 'bin') + os.pathsep + os.environ['PATH'], CALLS=str(calls),
                REAL_PYTHON=sys.executable, G0_FAIL=g0_fail, G1_EXIT=str(g1_exit), **env)
    done = subprocess.run([BASH, f'scripts/freqcone/{script}'], cwd=workspace, env=full, capture_output=True, text=True,
                          timeout=120)
    return done, calls.read_text().splitlines()


def pilot(workspace, level, **kw):
    decision = workspace / 'runs' / 'profile_scars' / 'decision.json'
    decision.write_text(json.dumps({'level': level, 'coarse_res_flag': 0}))
    return execute(workspace, 'pilot_cars.sh', **kw)


def events(lines):
    """The script's steps in order: runs, gates, selection."""
    out = []
    for line in lines:
        run = re.search(r'train\.train_HypSelEx .*--run_id (\S+)', line)
        gate = re.search(r'pilot_tools g0 runs/(\S+)', line)
        if run:
            out.append(run.group(1))
        elif gate:
            out.append(f'G0 {gate.group(1)}')
        elif 'select_pilot' in line:
            out.append('select')
        elif 'compare_runs' in line:
            name = re.search(r'--gate (\w+)', line)
            out.append(f'gate {name.group(1)}' if name else 'compare')
    return out


def run_lines(lines):
    return {re.search(r'--run_id (\S+)', x).group(1): x for x in lines if 'train.train_HypSelEx' in x}


@needs_bash
def test_level_a_runs_in_the_planned_order(workspace):
    done, lines = pilot(workspace, 'A')
    assert done.returncode == 0, done.stdout + done.stderr
    assert events(lines) == ['B8', 'G0 B8', 'R0', 'R1', 'R2', 'R3', 'select', 'gate g1', 'B9', 'B10', 'F8', 'F9', 'F10',
                             'gate final']
    runs = run_lines(lines)
    assert len(runs) == 10
    for rid in ('R0', 'R1', 'R2', 'R3'):
        assert '--split_train_val True' in runs[rid] and '--seed 7 ' in runs[rid], rid
    for rid, seed in (('B8', 8), ('B9', 9), ('B10', 10), ('F8', 8), ('F9', 9), ('F10', 10)):
        assert '--split_train_val' not in runs[rid] and f'--seed {seed} ' in runs[rid], rid
    for rid in runs:
        assert '--fix_label_order True' in runs[rid] and '--eval_every_epoch True' in runs[rid], rid
        assert ('--freqcone True' in runs[rid]) == (rid[0] == 'F' or rid in ('R1', 'R2', 'R3')), rid
    for rid in ('F8', 'F9', 'F10'):                         # the configuration of selection.json
        assert '--w_ent 0.1 --aperture A1' in runs[rid], rid
    assert '--w_ent 0.03 --aperture A1' in runs['R1'] and '--w_ent 0.1 --aperture A3' in runs['R3']
    assert '--base runs/R0 --fc runs/R2 --gate g1' in next(x for x in lines if '--gate g1' in x)
    assert '--base runs/B8 runs/B9 runs/B10 --fc runs/F8 runs/F9 runs/F10 --gate final' in lines[-1]


@needs_bash
def test_level_b_runs_in_the_planned_order(workspace):
    done, lines = pilot(workspace, 'B')
    assert done.returncode == 0, done.stdout + done.stderr
    assert events(lines) == ['B8', 'G0 B8', 'F8', 'B9', 'F9', 'B10', 'F10', 'gate final']
    runs = run_lines(lines)
    assert len(runs) == 6 and not any('--split_train_val' in x for x in runs.values())
    for rid in ('F8', 'F9', 'F10'):                         # fixed, not tuned on SSB
        assert '--w_ent 0.03 --aperture A1' in runs[rid], rid


@needs_bash
def test_a_failed_g0_runs_upstream_as_is_then_goes_on(workspace):
    done, lines = pilot(workspace, 'A', g0_fail='B8')
    assert done.returncode == 0, done.stdout + done.stderr
    assert events(lines)[:5] == ['B8', 'G0 B8', 'R0-repro', 'G0 R0-repro', 'R0']
    repro = run_lines(lines)['R0-repro']
    assert '--seed 1 ' in repro
    assert '--fix_label_order' not in repro and '--split_train_val' not in repro and '--freqcone True' not in repro
    assert 'the gap of runs/B8 comes from the label-order fix' in (workspace / 'runs' / 'g0_note.txt').read_text()


@needs_bash
def test_a_failed_repro_stops_before_the_pilot(workspace):
    done, lines = pilot(workspace, 'A', g0_fail='B8 R0-repro')
    assert done.returncode == 2
    assert events(lines) == ['B8', 'G0 B8', 'R0-repro', 'G0 R0-repro']


@needs_bash
def test_a_failed_g1_stops_before_the_confirmation(workspace):
    done, lines = pilot(workspace, 'A', g1_exit=1)
    assert done.returncode == 4
    assert events(lines)[-2:] == ['select', 'gate g1'] and 'B9' not in events(lines)


@needs_bash
def test_smoke_covers_both_data_modes(workspace):
    done, lines = execute(workspace, 'smoke_gpu.sh', SMOKE='1')
    assert done.returncode == 0, done.stdout + done.stderr
    runs = run_lines(lines)
    assert set(runs) == {'smoke_base', 'R1', 'R3', 'smoke_v3', 'B8', 'F8'}
    assert all('--smoke_frac 10' in x and '--fix_label_order True' in x for x in runs.values())
    for rid, x in runs.items():
        assert ('--split_train_val True' in x) == (rid not in ('B8', 'F8')), rid
    assert '--coarse_res 112' in runs['R3'] and '--coarse_res' not in runs['R1']
    assert any('compare_runs --base runs/smoke_scars/B8 --fc runs/smoke_scars/F8' in x for x in lines)
