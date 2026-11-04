"""Shared test helpers: a tiny CPU model with HypCD's own head and mapping, FreqCone arguments, a training loop
shaped like upstream's train() (same hook order), and synthetic run directories."""
import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from freqcone.hypcd import SCHEMA, add_args
from hyptorch.nn import ToPoincare
from models.vision_transformer import DINOHead


class TinyBackbone(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.conv = nn.Conv2d(3, dim, 5, stride=2, padding=2)
        self.lin = nn.Linear(dim, dim)

    def forward(self, x):
        return self.lin(torch.relu(self.conv(x)).mean((2, 3)))


def fc_args(**overrides):
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args([])
    base = dict(c=0.1, cr=2.0, batch_size=6, num_workers=0, seed=7, num_labeled_classes=3, dataset_name='scars',
                epochs=1, fc_f_a=2.0, fc_f_b=5.0, fc_log_every=1, train_classes=[0, 1, 2], unbalanced=False,
                image_size=32, dino='v2')
    for k, v in {**base, **overrides}.items():
        setattr(args, k, v)
    return args


def make_model(args, seed=0):
    torch.manual_seed(seed)
    model = TinyBackbone()
    head = DINOHead(in_dim=16, out_dim=64, nlayers=3, hidden_dim=32, bottleneck_dim=16)
    projection_head = nn.Sequential(head, ToPoincare(c=args.c, ball_dim=64, riemannian=False, clip_r=args.cr))
    return model, projection_head


class MergedTest(torch.utils.data.Dataset):
    """Stand-in for upstream's merge_train_loader_test dataset: (image, label, uq_idx, labelled flag)."""

    def __init__(self, images, labels, labelled):
        self.images, self.labels = images, labels
        self.labelled_dataset = labelled

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.images[i], int(self.labels[i]), i, 0


def noisy_test(model, loader, args):
    """Fake upstream test(): consumes every random generator, as k-means initialisation and loaders do."""
    np.random.rand(50)
    torch.rand(50)
    random.random()
    for _ in DataLoader(TensorDataset(torch.zeros(10, 1)), batch_size=3, shuffle=True):
        pass
    with torch.no_grad():
        s = float(sum(p.sum() for p in model.parameters()))
    return {'all_acc': 0.5 + 1e-6 * (s % 1), 'old_acc': 0.6, 'new_acc': 0.4}


def train_loop(fc, epochs, seed=0, stop_after=None):
    """Upstream's loop order: resume_skip / start_epoch -> steps -> end_epoch -> scheduler -> epoch_done."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    args = fc.args
    model, head = make_model(args, seed)
    opt = torch.optim.SGD(list(head.parameters()) + list(model.parameters()), lr=0.05, momentum=0.9)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    g = torch.Generator().manual_seed(123)
    data = TensorDataset(torch.randn(24, 3, 32, 32, generator=g), torch.arange(24) % 6)
    loader = DataLoader(data, batch_size=6, sampler=RandomSampler(data), drop_last=True)
    merged = DataLoader(MergedTest(data.tensors[0], data.tensors[1], TensorDataset(data.tensors[0][:9],
                                                                                torch.arange(9) % 3)), batch_size=6)
    for epoch in range(epochs):
        if fc.resume_skip(epoch, model, head, opt, sched):
            continue
        fc.start_epoch(epoch)
        for i, (x, _) in enumerate(loader):
            x2 = x + 0.05 * torch.from_numpy(np.random.randn(*x.shape)).float()     # numpy-driven augmentation
            images = torch.cat([x, x2])
            ball = head(model(images))
            loss = 1e-2 * ball.pow(2).sum()
            loss_fc = fc.step(model, head[0], images, ball, x.shape[0], epoch, i)
            if loss_fc is not None:
                loss = loss + loss_fc
            fc.check_finite(loss, epoch, i)
            opt.zero_grad()
            loss.backward()
            opt.step()
        fc.end_epoch(epoch, float(loss), float(loss), 0.0)
        sched.step()
        fc.epoch_done(epoch, len(loader), model, head, merged, noisy_test, (0.5, 0.6, 0.4), opt, sched)
        if stop_after is not None and epoch + 1 == stop_after:
            return model, head, merged
    return model, head, merged


def write_run(root, run_id, seed, enable, curves, val_acc=None, w_ent=0.03, aperture='A1', variant='v2',
              val=None, epochs=None, upstream_old=None):
    """A synthetic run directory: run.json (no test numbers) and the two JSONL files.
    curves: dict all/old/new -> list per epoch (fractions)."""
    d = os.path.join(root, run_id)
    os.makedirs(d, exist_ok=True)
    n = len(curves['all']) if epochs is None else epochs
    rec = {'schema': SCHEMA, 'run_id': run_id, 'seed': seed, 'dataset': 'scars', 'base_method': 'Hyp-SelEx',
           'c': 0.1, 'r': 2.0, 'epochs': n, 'fix_label_order': True, 'split_train_val': True,
           'freqcone': {'enable': enable, 'variant': variant if enable else None, 'w_ent': w_ent, 'aperture': aperture},
           'val': val}
    with open(os.path.join(d, 'run.json'), 'w') as f:
        json.dump(rec, f)
    with open(os.path.join(d, 'epochs_test.jsonl'), 'w') as f:
        for e in range(n):
            up = upstream_old[e] if upstream_old else curves['old'][e]
            f.write(json.dumps({'epoch': e + 1, 'all': curves['all'][e], 'old': curves['old'][e], 'new': curves['new'][e],
                                'upstream_disjoint_test': {'all': 0.5, 'old': up, 'new': 0.5}}) + '\n')
    with open(os.path.join(d, 'epochs_val.jsonl'), 'w') as f:
        for e in range(n):
            f.write(json.dumps({'epoch': e + 1, 'val_acc': val_acc[e] if val_acc else None}) + '\n')
    return d
