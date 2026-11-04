"""FCRun hooks on a tiny CPU model with HypCD's own head (DINOHead) and mapping (ToPoincare)."""
import argparse
import json
import math
import types

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from conftest import FakeKMeans, needs_cuda
from freqcone.hypcd import FCRun, add_args
from freqcone.pilot_tools import check_record
from hyptorch.nn import ToPoincare
from models.vision_transformer import DINOHead


class TinyBackbone(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.conv = nn.Conv2d(3, dim, 5, stride=2, padding=2)
        self.lin = nn.Linear(dim, dim)

    def forward(self, x):
        return self.lin(torch.relu(self.conv(x)).mean((2, 3)))


def make(seed=0, **overrides):
    torch.manual_seed(seed)
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args([])
    base = dict(c=0.1, cr=1.5, batch_size=6, num_workers=0, seed=7, num_labeled_classes=3, dataset_name='cub',
                epochs=1, fc_f_a=2.0, fc_f_b=5.0, fc_log_every=1, train_classes=[0, 1, 2], unbalanced=False,
                image_size=32, model_path='unused/model.pt')
    for k, v in {**base, **overrides}.items():
        setattr(args, k, v)
    model = TinyBackbone()
    head = DINOHead(in_dim=16, out_dim=64, nlayers=3, hidden_dim=32, bottleneck_dim=16)
    projection_head = nn.Sequential(head, ToPoincare(c=args.c, ball_dim=64, riemannian=False, clip_r=args.cr))
    return args, model, projection_head


def test_step_adds_a_finite_term_and_routes_gradients():
    args, model, projection_head = make(freqcone=True, fc_w_ent=1.0)
    run = FCRun(args, torch.device('cpu'))
    assert run.enabled and run.loss is not None
    run.start_epoch(0)
    B = 6
    images = torch.randn(2 * B, 3, 32, 32)
    ball = projection_head(model(images))
    ball.retain_grad()
    loss_fc = run.step(model, projection_head[0], images, ball, B, epoch=0, batch_idx=0)
    assert loss_fc is not None and torch.isfinite(loss_fc)
    loss_fc.backward()
    assert float(ball.grad[B:].abs().max()) == 0.0          # u2: no gradient from FreqCone
    assert float(ball.grad[:B].abs().max()) > 0.0           # u1: steered by L_ent / L_con
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in projection_head[0].mlp.parameters())
    assert model.conv.weight.grad.abs().sum() > 0           # coarse views train the backbone too
    diag = run.diagnostics[-1]
    assert diag['in_cone'] is not None and diag['angle_b_u'] is not None and diag['depth_full'] > 0
    run.end_epoch(0, 1.0, 1.0, 0.0)
    assert {'fc_dir', 'fc_ent', 'fc_con', 'fc_in_cone', 'fc_depth_full', 'fc_angle_b_u', 'clip_frac'} <= set(run.history[-1])


def test_baseline_logs_sharp_view_depth_without_a_term():
    args, model, projection_head = make(results_json='unused.json')
    run = FCRun(args, torch.device('cpu'))
    images = torch.randn(12, 3, 32, 32)
    ball = projection_head(model(images))
    run.start_epoch(0)
    assert run.enabled and run.step(model, projection_head[0], images, ball, 6, 0, 0) is None
    d = run.diagnostics[-1]
    assert d['depth_full'] > 0 and d['clip_frac'] is not None and d['in_cone'] is None


def test_depth_threshold_follows_c_and_r_b():
    for c, ap, expected in ((0.1, 'A1', 1.282), (0.1, 'A2', 0.866), (0.1, 'A3', 0.724), (0.05, 'A1', 1.813)):
        run = FCRun(make(c=c, cr=1.5, fc_aperture=ap)[0], torch.device('cpu'))
        assert abs(run.depth_b - expected) < 1e-3
    run = FCRun(make(fc_r_a=0.11, fc_r_b=0.15)[0], torch.device('cpu'))
    assert run.aperture_name == 'custom'
    with pytest.raises(ValueError):
        FCRun(make(fc_r_a=0.11)[0], torch.device('cpu'))


def test_cutoff_checks():
    with pytest.raises(ValueError, match='fc_f_a < fc_f_b'):
        FCRun(make(fc_f_a=5.0, fc_f_b=5.0)[0], torch.device('cpu'))
    with pytest.raises(ValueError, match='aliasing'):
        FCRun(make(fc_f_a=2.0, fc_f_b=8.0, fc_coarse_size=16)[0], torch.device('cpu'))    # needs f_b < 8
    FCRun(make(fc_f_a=2.0, fc_f_b=7.5, fc_coarse_size=16)[0], torch.device('cpu'))


def test_coarse_views_at_half_size():
    args, model, projection_head = make(freqcone=True, fc_coarse_size=16)
    run = FCRun(args, torch.device('cpu'))
    run.start_epoch(0)
    images = torch.randn(12, 3, 32, 32)
    assert torch.isfinite(run.step(model, projection_head[0], images, projection_head(model(images)), 6, 0, 0))


def test_non_finite_loss_stops_the_run():
    run = FCRun(make(freqcone=True)[0], torch.device('cpu'))
    run.check_finite(torch.tensor(1.0), 0, 0)
    with pytest.raises(FloatingPointError, match='do not lower lambda'):
        run.check_finite(torch.tensor(float('nan')), 3, 41)


def test_profile_epoch_estimate(tmp_path):
    out = tmp_path / 'profile.json'
    args, _, _ = make(profile_steps=3, results_json=str(out), run_id='profile', run_tag='fc224')
    run = FCRun(args, torch.device('cpu'))
    run.start_epoch(0)
    assert [run.profile_tick(i) for i in range(3)] == [False, False, True]
    run.end_epoch(0, 1.0, 1.0, 0.0)
    assert run.epoch_done(0, steps_per_epoch=40) is True
    rec = json.loads(out.read_text())
    p = rec['profile']
    assert p['steps_trained'] == 3 and p['steps_per_epoch'] == 40
    assert abs(p['est_epoch_s'] - (p['epoch_wall_s_measured'] + 37 * p['s_per_step_median'])) < 1e-9
    assert check_record(rec, profile=True) == []


class FakeSplit:
    """Labelled / unlabelled / validation subsets with uq_idxs and pandas-free targets (Stanford Cars layout)."""

    def __init__(self, idx, targets):
        self.uq_idxs, self.data, self.target = list(idx), [f'/x/{i}.jpg' for i in idx], [t + 1 for t in targets]


def test_final_record_has_the_full_schema(tmp_path):
    out = tmp_path / 'R2.json'
    args, model, projection_head = make(freqcone=True, results_json=str(out), split_train_val=True, run_id='R2',
                                        fc_diag_images=0, model_path=str(tmp_path / 'model.pt'))
    g = torch.Generator().manual_seed(1)
    # training subsets and validation split (disjoint uq_idxs; validation from known classes 0-2 only)
    train = types.SimpleNamespace(labelled_dataset=FakeSplit(range(0, 30), [i % 3 for i in range(30)]),
                                  unlabelled_dataset=FakeSplit(range(30, 60), [i % 6 for i in range(30)]))
    val_split = FakeSplit(range(60, 75), [i % 3 for i in range(15)])
    images = torch.randn(75, 3, 32, 32, generator=g)
    val_ds = TensorDataset(images[60:], torch.arange(15) % 3)
    run = FCRun(args, torch.device('cpu'), train_dataset=train, val_dataset=val_split)
    assert run.split['n_val'] == 15
    run.val_dataset = val_ds                 # the same 15 images as tensors, for the feature pass
    run.kmeans_cls = FakeKMeans
    run.start_epoch(0)
    x = images[:12]
    run.step(model, projection_head[0], torch.cat([x[:6], x[6:]]), projection_head(model(x)), 6, 0, 0)
    run.end_epoch(0, 1.0, 1.0, 0.5)
    run.epoch_done(0, 10)
    # "best" checkpoint differs from the final weights; final weights must be restored afterwards
    final = {k: v.clone() for k, v in model.state_dict().items()}
    best = {k: v + 1.0 for k, v in final.items()}
    torch.save(best, tmp_path / 'model_best.pt')

    class Merged(torch.utils.data.Dataset):
        labelled_dataset = TensorDataset(images[:30], torch.arange(30) % 3)

        def __len__(self):
            return 24

        def __getitem__(self, i):
            return images[30 + i], i % 6, 30 + i, 0
    loader = DataLoader(Merged(), batch_size=6)

    def fake_test(m, ldr, a):
        w = float(m.lin.bias.sum())
        return {'all_acc': 0.5 + 0.0 * w, 'old_acc': 0.6, 'new_acc': 0.4, 'w': w}
    seen = []
    res = run.final_eval(model, projection_head, loader, lambda m, l, a: seen.append(float(m.lin.bias.sum())) or
                         fake_test(m, l, a), best_epoch=0, best_stats=[0.4, 0.5, 0.3])
    assert abs(seen[1] - (seen[0] + 16.0)) < 1e-4                       # upstream_best was scored on the best weights
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in final.items())    # final weights restored
    rec = json.loads(out.read_text())
    assert check_record(rec) == [], check_record(rec)
    assert rec['test_last']['all'] == 0.5 and rec['upstream_best']['epoch'] == 0
    assert rec['val']['metric'] == 'ss_kmeans_known_hungarian' and len(rec['val']['per_sample_pred']) == 15
    assert rec['val']['val_list_sha256'] == run.split['val_list_sha256']
    geo = rec['geometry_final']
    assert geo['n_images'] == 24 and set(geo['per_aperture']) == {'A1', 'A2', 'A3'}
    assert abs(geo['per_aperture']['A2']['depth_b'] - 0.866) < 1e-3
    fc = rec['freqcone']
    assert fc['aperture'] == 'A1' and abs(fc['psi_a'] - 55.2) < 0.2 and abs(fc['depth_b'] - 1.282) < 1e-3
    assert res['cost']['time_per_epoch_s'] is not None


@needs_cuda
def test_val_metric_with_upstream_kmeans_on_cuda():
    args, _, _ = make(split_train_val=False, num_labeled_classes=4)
    run = FCRun(args, torch.device('cuda'))
    g = torch.Generator().manual_seed(0)
    centers = torch.randn(4, 8, generator=g) * 5
    lab_y, val_y = torch.arange(4).repeat(10), torch.arange(4).repeat(3)
    lab = TensorDataset(centers[lab_y] + torch.randn(40, 8, generator=g), lab_y)
    run.val_dataset = TensorDataset(centers[val_y] + torch.randn(12, 8, generator=g), val_y)
    run.split = {'val_list_sha256': 'x'}
    out = run.val_metric(nn.Identity().cuda(), lab)
    assert out['value'] > 0.9 and len(out['per_sample_pred']) == 12
