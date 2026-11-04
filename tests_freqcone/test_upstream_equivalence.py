"""With the default flags the patched code base computes exactly what upstream HypCD computes.

1. Every change to an upstream file is an inserted block between ``# >>> freqcone`` and
   ``# <<< freqcone``: deleting those blocks gives back the upstream file byte for byte
   (line endings normalised, since git may check files out with CRLF).
2. Every inserted block is inactive with the default flags (FCRun.enabled is False, the distance
   wrapper calls the upstream function), checked below.
"""
import argparse
import importlib
import os
import re
import subprocess
import sys
import types

import pytest
import torch

from conftest import ROOT

UPSTREAM_COMMIT = 'ffc895b7ff38be8502bb7e3bd851f1d2a55dff73'
PATCHED_FILES = ['train/train_HypSelEx.py', 'data/get_datasets.py', 'hyptorch/pmath.py']
START, END = re.compile(r'^\s*# >>> freqcone'), re.compile(r'^\s*# <<< freqcone')


def strip_blocks(text):
    out, inside = [], False
    for line in text.split('\n'):
        if not inside and START.match(line):
            inside = True
            continue
        if inside:
            if END.match(line):
                inside = False
            continue
        out.append(line)
    assert not inside, 'unterminated # >>> freqcone block'
    return '\n'.join(out)


def upstream_text(path):
    try:
        res = subprocess.run(['git', 'show', f'{UPSTREAM_COMMIT}:{path}'], cwd=ROOT, capture_output=True, timeout=60)
    except FileNotFoundError:
        pytest.skip('git not available')
    if res.returncode != 0:
        pytest.skip(f'upstream commit not available: {res.stderr.decode()[:200]}')
    return res.stdout.decode('utf-8').replace('\r\n', '\n')


@pytest.mark.parametrize('path', PATCHED_FILES)
def test_only_marked_insertions(path):
    with open(os.path.join(ROOT, path), encoding='utf-8', newline='') as f:
        patched = f.read().replace('\r\n', '\n')
    assert '# >>> freqcone' in patched, f'{path} has no FreqCone block (nothing to check)'
    assert strip_blocks(patched) == upstream_text(path)


def fc_args(**overrides):
    from freqcone.hypcd import add_args
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args([])
    base = dict(c=0.05, cr=2.3, batch_size=4, num_workers=0, seed=0, num_labeled_classes=3, dataset_name='toy',
                epochs=1)
    for k, v in {**base, **overrides}.items():
        setattr(args, k, v)
    return args


def test_defaults_leave_every_hook_inactive():
    from freqcone.hypcd import FCRun
    args = fc_args()
    assert (args.dist_impl, args.freqcone, args.fix_label_order, args.split_train_val, args.smoke_frac,
            args.run_dir, args.profile_steps, args.eval_every_epoch, args.hide_test_stdout, args.resume,
            args.freqcone_variant) == ('upstream', False, False, False, 0, '', 0, False, False, False, 'v2')
    run = FCRun(args, torch.device('cpu'), val_dataset=None)
    assert run.enabled is False and run.loss is None and run.split is None and run.profiling is False
    assert run.run_dir is None and run.resume_state is None


@pytest.fixture(scope='module')
def train_module():
    sys.argv = ['pytest']
    return importlib.import_module('train.train_HypSelEx')


def ball_points(n, d, seed, dtype=torch.float32):
    from hyptorch.nn import ToPoincare
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, d, generator=g, dtype=dtype) * torch.rand(n, 1, generator=g, dtype=dtype) * 4
    return ToPoincare(c=0.05, clip_r=2.3, riemannian=False)(z)


def test_distance_wrapper_is_upstream_by_default(train_module):
    from hyptorch.pmath import dist_matrix as upstream
    assert train_module.DIST_IMPL == 'upstream'
    x, y = ball_points(24, 64, 0), ball_points(20, 64, 1)
    assert torch.equal(train_module.dist_matrix(x, y, c=0.05), upstream(x, y, c=0.05))


def test_inert_hooks_object_by_default(train_module):
    assert train_module.FC_RUN.enabled is False


def test_base_losses_lowmem_match_upstream(train_module):
    """SupConLoss and info_nce_logits of the training script give the same loss and gradients either way."""
    m = train_module
    m.device = torch.device('cpu')
    m.args = types.SimpleNamespace(c=0.05, n_views=2, temperature=1.0)
    B, D = 16, 128
    results = {}
    for impl in ('upstream', 'lowmem'):
        m.DIST_IMPL = impl
        try:
            z = ball_points(2 * B, D, 3, dtype=torch.float64).requires_grad_(True)
            labels = torch.arange(B) % 5
            sup = m.SupConLoss(hyper_c=0.05)(torch.stack([z[:B], z[B:]], 1), labels=labels)
            conf = torch.rand(2 * B, 2 * B, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
            logits, lab, sim = m.info_nce_logits(z, conf, m.args, hyper_c=0.05)
            nce = m.LabelSmoothingLoss()(logits, lab, sim, 1.0)
            (sup + nce).backward()
            results[impl] = (sup.detach(), nce.detach(), z.grad.clone())
        finally:
            m.DIST_IMPL = 'upstream'
    (s0, n0, g0), (s1, n1, g1) = results['upstream'], results['lowmem']
    assert torch.isfinite(g1).all()
    assert abs(float(s0 - s1)) < 1e-6 and abs(float(n0 - n1)) < 1e-6
    assert float((g0 - g1).abs().max() / g0.abs().max()) < 1e-5
