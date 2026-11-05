"""FCRun hooks on a tiny CPU model with HypCD's own head (DINOHead) and mapping (ToPoincare)."""
import json

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from conftest import FakeKMeans, needs_cuda
from data.data_utils import MergedDataset
from freqcone.hypcd import FCRun
from freqcone.pilot_tools import check_run_dir
from helpers import Split, fc_args, make_model


def test_step_adds_a_finite_term_and_routes_gradients():
    args = fc_args(freqcone=True, fc_w_ent=1.0)
    model, projection_head = make_model(args)
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


def test_variant_and_flag_aliases():
    from freqcone.hypcd import add_args
    import argparse
    p = argparse.ArgumentParser()
    add_args(p)
    a = p.parse_args(['--freqcone_variant', 'v3', '--w_ent', '0.1', '--aperture', 'A3', '--coarse_res', '112'])
    b = p.parse_args(['--freqcone_variant', 'v3', '--fc_w_ent', '0.1', '--fc_aperture', 'A3', '--fc_coarse_size', '112'])
    assert vars(a) == vars(b) and (a.fc_w_ent, a.fc_aperture, a.fc_coarse_size) == (0.1, 'A3', 112)
    run = FCRun(fc_args(freqcone=True, freqcone_variant='v3'), torch.device('cpu'))
    assert run.loss.detach_child_norm is False


def test_baseline_logs_sharp_view_depth_without_a_term():
    args = fc_args(run_dir='')
    args.profile_steps = 0
    model, projection_head = make_model(args)
    run = FCRun(fc_args(freqcone=False, profile_steps=0, fc_log_every=1, run_dir=''), torch.device('cpu'))
    images = torch.randn(12, 3, 32, 32)
    ball = projection_head(model(images))
    run.start_epoch(0)
    assert run.step(model, projection_head[0], images, ball, 6, 0, 0) is None
    d = run.diagnostics[-1]
    assert d['depth_full'] > 0 and d['clip_frac'] is not None and d['in_cone'] is None


def test_depth_threshold_follows_c_and_r_b():
    for c, ap, expected in ((0.1, 'A1', 1.282), (0.1, 'A2', 0.866), (0.1, 'A3', 0.724), (0.05, 'A1', 1.813)):
        run = FCRun(fc_args(c=c, cr=1.5, fc_aperture=ap), torch.device('cpu'))
        assert abs(run.depth_b - expected) < 1e-3
    assert FCRun(fc_args(fc_r_a=0.11, fc_r_b=0.15), torch.device('cpu')).aperture_name == 'custom'
    with pytest.raises(ValueError):
        FCRun(fc_args(fc_r_a=0.11), torch.device('cpu'))


def test_cutoff_checks():
    with pytest.raises(ValueError, match='fc_f_a < fc_f_b'):
        FCRun(fc_args(fc_f_a=5.0, fc_f_b=5.0), torch.device('cpu'))
    with pytest.raises(ValueError, match='aliasing'):
        FCRun(fc_args(fc_f_a=2.0, fc_f_b=8.0, fc_coarse_size=16), torch.device('cpu'))    # needs f_b < 8
    FCRun(fc_args(fc_f_a=2.0, fc_f_b=7.5, fc_coarse_size=16), torch.device('cpu'))


def test_coarse_views_at_half_size():
    args = fc_args(freqcone=True, fc_coarse_size=16)
    model, projection_head = make_model(args)
    run = FCRun(args, torch.device('cpu'))
    run.start_epoch(0)
    images = torch.randn(12, 3, 32, 32)
    assert torch.isfinite(run.step(model, projection_head[0], images, projection_head(model(images)), 6, 0, 0))


def test_non_finite_loss_stops_the_run():
    run = FCRun(fc_args(freqcone=True), torch.device('cpu'))
    run.check_finite(torch.tensor(1.0), 0, 0)
    with pytest.raises(FloatingPointError, match='do not lower lambda'):
        run.check_finite(torch.tensor(float('nan')), 3, 41)


def test_final_record_with_validation(tmp_path):
    args = fc_args(freqcone=True, run_dir=str(tmp_path / 'R1'), split_train_val=True, run_id='R1', fc_diag_images=0)
    model, projection_head = make_model(args)
    g = torch.Generator().manual_seed(1)
    images = torch.randn(75, 3, 32, 32, generator=g)
    train = MergedDataset(Split(images[:30], [i % 3 for i in range(30)], range(0, 30)),
                          Split(images[30:60], [i % 6 for i in range(30)], range(30, 60)))
    val_split = Split(images[60:], [i % 3 for i in range(15)], range(60, 75))
    run = FCRun(args, torch.device('cpu'), train_dataset=train, val_dataset=val_split,
                datasets={'train_labelled': train.labelled_dataset, 'val': val_split})
    assert run.split['n_val'] == 15 and run.data_lists['val']['n'] == 15
    run.kmeans_cls = FakeKMeans
    run.start_epoch(0)
    x = images[:12]
    run.step(model, projection_head[0], x, projection_head(model(x)), 6, 0, 0)
    run.record_supcon(torch.ones(6, dtype=torch.bool), 1.0, 1.0)
    run.end_epoch(0, 1.0, 1.0, 0.5)
    loader = DataLoader(train, batch_size=6)
    run.epoch_done(0, 10, model, projection_head, loader, lambda m, l, a: {'all_acc': .5, 'old_acc': .6, 'new_acc': .4},
                   (0.5, 0.6, 0.4), gcd_used=(0.55, 0.65, 0.45))
    run.final_eval(model, projection_head, loader, lambda m, l, a: {'all_acc': .5, 'old_acc': .6, 'new_acc': .4})
    rec = json.loads((tmp_path / 'R1' / 'run.json').read_text())
    assert check_run_dir(str(tmp_path / 'R1')) == []
    assert rec['val']['metric'] == 'ss_kmeans_known_hungarian' and len(rec['val']['per_sample_pred']) == 15
    assert rec['val']['val_list_sha256'] == run.split['val_list_sha256']
    geo = rec['geometry_final']
    assert geo['n_images'] == 60 and set(geo['per_aperture']) == {'A1', 'A2', 'A3'}
    assert abs(geo['per_aperture']['A2']['depth_b'] - 0.866) < 1e-3
    fc = rec['freqcone']
    assert fc['aperture'] == 'A1' and abs(fc['psi_a'] - 55.2) < 0.2 and abs(fc['depth_b'] - 1.282) < 1e-3
    test_line = json.loads((tmp_path / 'R1' / 'epochs_test.jsonl').read_text().splitlines()[-1])
    assert test_line == {'epoch': 1, 'disjoint_test': {'all': 0.5, 'old': 0.6, 'new': 0.4},
                         'gcd': {'all': 0.55, 'old': 0.65, 'new': 0.45, 'source': 'ss_kmeans_epoch_start',
                                 'weights_after_epoch': 0},
                         'eval': {'all': 0.5, 'old': 0.6, 'new': 0.4}}


@needs_cuda
def test_val_metric_with_upstream_kmeans_on_cuda():
    args = fc_args(num_labeled_classes=4)
    run = FCRun(args, torch.device('cuda'))
    g = torch.Generator().manual_seed(0)
    centers = torch.randn(4, 8, generator=g) * 5
    lab_y, val_y = torch.arange(4).repeat(10), torch.arange(4).repeat(3)
    lab = TensorDataset(centers[lab_y] + torch.randn(40, 8, generator=g), lab_y)
    run.val_dataset = TensorDataset(centers[val_y] + torch.randn(12, 8, generator=g), val_y)
    run.split = {'val_list_sha256': 'x'}
    out = run.val_metric(nn.Identity().cuda(), lab, random_state=7)
    assert out['value'] > 0.9 and len(out['per_sample_pred']) == 12
