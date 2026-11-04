"""Reading run directories (runs/<run_id>/) and the selection lock shared by the reporting scripts.

A run directory holds run.json (no test number), epochs_val.jsonl and epochs_test.jsonl. Test numbers of the
pilot candidates (R1-R3) may be read only once RUNS_ROOT/selection.json is committed, unchanged since, and every
input file it lists still has its recorded SHA-256 (check_locked).
"""
import hashlib
import json
import os
import subprocess

from freqcone.hypcd import REPO_ROOT, SCHEMA

CANDIDATES = ('R1', 'R2', 'R3')


class LockError(RuntimeError):
    pass


def utf8_stdout():
    """The reports are in Vietnamese; consoles with a legacy code page (Windows cp1252) cannot print them."""
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def read_run_json(run_dir):
    with open(os.path.join(run_dir, 'run.json')) as f:
        rec = json.load(f)
    if rec.get('schema') != SCHEMA:
        raise ValueError(f"{run_dir}: schema {rec.get('schema')!r}, expected {SCHEMA}")
    return rec


def check_locked(selection_path, repo=REPO_ROOT):
    """Raise LockError unless the selection is committed, unchanged, and its inputs unchanged."""
    rel = os.path.relpath(os.path.abspath(selection_path), os.path.abspath(repo)).replace('\\', '/')
    if subprocess.run(['git', 'ls-files', '--error-unmatch', rel], cwd=repo, capture_output=True).returncode:
        raise LockError(f'{rel} is not committed: commit the selection before looking at test numbers of R1-R3')
    if subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', rel], cwd=repo).returncode:
        raise LockError(f'{rel} changed after its commit')
    with open(selection_path) as f:
        sel = json.load(f)
    root = os.path.dirname(os.path.abspath(selection_path))
    for name, digest in sel['inputs_sha256'].items():
        with open(os.path.join(root, name), 'rb') as f:
            if hashlib.sha256(f.read()).hexdigest() != digest:
                raise LockError(f'{name} changed after the selection')
    return sel


def load_run(run_dir, with_test=False, repo=REPO_ROOT):
    """run.json + epochs_val.jsonl (+ epochs_test.jsonl, behind the lock for R1-R3)."""
    rec = read_run_json(run_dir)
    out = {'dir': run_dir, 'run': rec, 'val': read_jsonl(os.path.join(run_dir, 'epochs_val.jsonl'))}
    if with_test:
        if rec.get('run_id') in CANDIDATES:
            check_locked(os.path.join(os.path.dirname(os.path.abspath(run_dir)), 'selection.json'), repo)
        out['test'] = read_jsonl(os.path.join(run_dir, 'epochs_test.jsonl'))
    return out
