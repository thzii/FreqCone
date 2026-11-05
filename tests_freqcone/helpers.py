"""Shared test helpers: a tiny CPU model with HypCD's own head and mapping, FreqCone arguments, a training loop
shaped like upstream's train() (same hook order), the training script's data path on in-memory Stanford Cars, and
synthetic run directories."""
import argparse
import copy
import importlib
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, TensorDataset, WeightedRandomSampler

from conftest import FakeCars
from data.data_utils import MergedDataset
from freqcone.hypcd import SCHEMA, FCRun, add_args
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


class Split(torch.utils.data.Dataset):
    """A data subset in the Stanford Cars layout (path list `data`, 1-based `target`, `uq_idxs`), so that the real
    MergedDataset and the data checks work on it. Items: (image, 0-based class, uq_idx)."""

    def __init__(self, images, labels, uq_idxs):
        self.images, self.uq_idxs = images, np.asarray(uq_idxs)
        self.data = [f'/x/{i}.jpg' for i in uq_idxs]
        self.target = [int(t) + 1 for t in labels]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.images[i], self.target[i] - 1, int(self.uq_idxs[i])


def merged_loader(images, labels, n_labelled, batch_size=6):
    """Upstream's merged loaders on tensors: real MergedDataset, the first n_labelled images labelled."""
    n = len(images)
    return DataLoader(MergedDataset(Split(images[:n_labelled], labels[:n_labelled], range(n_labelled)),
                                    Split(images[n_labelled:], labels[n_labelled:], range(n_labelled, n))),
                      batch_size=batch_size)


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
    merged = merged_loader(data.tensors[0], torch.cat([torch.arange(9) % 3, torch.arange(9, 24) % 6]), 9)
    if fc.data_counts is None:
        fc.data_counts = fc._prepare_training_data(merged.dataset, {})
    for epoch in range(epochs):
        if fc.resume_skip(epoch, model, head, opt, sched):
            continue
        fc.start_epoch(epoch)
        for i, (x, _) in enumerate(loader):
            x2 = x + 0.05 * torch.from_numpy(np.random.randn(*x.shape)).float()     # numpy-driven augmentation
            images = torch.cat([x, x2])
            ball = head(model(images))
            loss = 1e-2 * ball.pow(2).sum()
            fc.record_supcon(torch.ones(x.shape[0], dtype=torch.bool), 1.0, 1.0)
            loss_fc = fc.step(model, head[0], images, ball, x.shape[0], epoch, i)
            if loss_fc is not None:
                loss = loss + loss_fc
            fc.check_finite(loss, epoch, i)
            opt.zero_grad()
            loss.backward()
            opt.step()
        fc.end_epoch(epoch, float(loss), float(loss), 0.0)
        sched.step()
        fc.epoch_done(epoch, len(loader), model, head, merged, noisy_test, (0.5, 0.6, 0.4), opt, sched,
                      gcd_used=(0.55, 0.65, 0.45))
        if stop_after is not None and epoch + 1 == stop_after:
            return model, head, merged
    return model, head, merged


class NoWriter:
    def add_scalar(self, *a, **k):
        pass

    def add_scalars(self, *a, **k):
        pass


def jitter(image):
    return image + 0.05 * torch.randn_like(image)


def build_upstream(tmp_path, monkeypatch, device, epochs, split=True, smoke_frac=0, batch_size=16, **over):
    """The data path of train/train_HypSelEx.py on FakeCars: upstream's get_scars_datasets and get_datasets, the
    script's sampler and five loaders (its lines are under __main__, so they are repeated here), and FCRun built as
    the script builds it. Returns the module, args, model, head, loaders and the datasets dict."""
    import data.stanford_cars
    from data.get_datasets import get_datasets
    monkeypatch.setattr(sys, 'argv', ['pytest'])
    monkeypatch.setattr(data.stanford_cars, 'CarsDataset', FakeCars)
    T = importlib.import_module('train.train_HypSelEx')
    known, novel = list(range(FakeCars.N_KNOWN)), list(range(FakeCars.N_KNOWN, FakeCars.N_CLASSES))
    ns = argparse.Namespace(dataset_name='scars', train_classes=known, unlabeled_classes=novel, prop_train_labels=0.5,
                            split_train_val=split, smoke_frac=smoke_frac)
    train_view = T.ContrastiveLearningViewGenerator(base_transform=jitter, n_views=2)
    train_dataset, test_dataset, unlabelled_test, datasets = get_datasets('scars', train_view, None, ns)
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / (unlabelled_len + label_len) for i in range(len(train_dataset))]
    sampler = WeightedRandomSampler(torch.DoubleTensor(sample_weights), num_samples=len(train_dataset))
    train_dataset_test = copy.deepcopy(train_dataset)
    train_dataset_test.labelled_dataset.transform = None
    train_dataset_test.unlabelled_dataset.transform = None
    loaders = dict(
        train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=sampler, drop_last=True),
        test_loader=DataLoader(test_dataset, batch_size=batch_size // 2, shuffle=False),
        unlabelled_train_loader=DataLoader(unlabelled_test, batch_size=batch_size // 2, shuffle=False),
        merge_train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=False),
        merge_train_loader_test=DataLoader(train_dataset_test, batch_size=batch_size, shuffle=False))
    settings = dict(run_dir=str(tmp_path / 'run'), epochs=epochs, num_labeled_classes=len(known),
                    num_unlabeled_classes=len(novel), train_classes=known, unlabeled_classes=novel,
                    batch_size=batch_size, lr=0.01, momentum=0.9, weight_decay=0.0, strategy='zero_one',
                    cluster_momentum=1.0, unsupervised_smoothing=1.0, train_report_interval=epochs,
                    prototype_extraction_interval=1, distance='euclidean', contrast_unlabel_only=False,
                    sup_con_weight=0.35, n_views=2, temperature=1.0, hyper_start_epoch=0, hyper_end_epoch=epochs,
                    hyper_max_weight=1.0, writer=NoWriter(), model_path=str(tmp_path / 'model.pt'), eval_funcs=['v2'],
                    c=0.1, cr=2.0, split_train_val=split, smoke_frac=smoke_frac, fix_label_order=split)
    args = fc_args(**{**settings, **over})
    torch.manual_seed(0)
    model, head = make_model(args)
    model, head = model.to(device), head.to(device)
    run = FCRun(args, device, train_dataset=train_dataset, val_dataset=datasets.get('val'), datasets=datasets,
                train_copies=(train_dataset_test, unlabelled_test))
    for name, value in (('args', args), ('device', device), ('FC_RUN', run)):
        monkeypatch.setattr(T, name, value, raising=False)      # restored afterwards: other tests share the module
    return T, args, model, head, loaders, datasets


def run_train(T, args, model, head, loaders):
    T.train(head, model, loaders['train_loader'], loaders['test_loader'], loaders['unlabelled_train_loader'],
            loaders['merge_train_loader'], loaders['merge_train_loader_test'], args)


def write_run(root, run_id, seed, enable, curves, val_acc=None, w_ent=0.03, aperture='A1', variant='v2',
              val=None, epochs=None, disjoint_old=None, gcd=None, split_train_val=True):
    """A synthetic run directory: run.json (no test numbers) and the two JSONL files.
    curves: all/old/new per epoch of the per-epoch evaluation (`eval`, fractions); gcd: same layout for upstream's
    training-set numbers (default: curves); disjoint_old: Old on the disjoint test split (default: curves['old'])."""
    d = os.path.join(root, run_id)
    os.makedirs(d, exist_ok=True)
    n = len(curves['all']) if epochs is None else epochs
    gcd = gcd or curves
    rec = {'schema': SCHEMA, 'run_id': run_id, 'seed': seed, 'dataset': 'scars', 'base_method': 'Hyp-SelEx',
           'c': 0.1, 'r': 2.0, 'epochs': n, 'fix_label_order': True, 'split_train_val': split_train_val,
           'freqcone': {'enable': enable, 'variant': variant if enable else None, 'w_ent': w_ent, 'aperture': aperture},
           'val': val}
    with open(os.path.join(d, 'run.json'), 'w') as f:
        json.dump(rec, f)
    with open(os.path.join(d, 'epochs_test.jsonl'), 'w') as f:
        for e in range(n):
            dold = disjoint_old[e] if disjoint_old else curves['old'][e]
            f.write(json.dumps({'epoch': e + 1,
                                'gcd': {'all': gcd['all'][e], 'old': gcd['old'][e], 'new': gcd['new'][e],
                                        'source': 'ss_kmeans_epoch_start', 'weights_after_epoch': e},
                                'disjoint_test': {'all': 0.5, 'old': dold, 'new': 0.5},
                                'eval': {'all': curves['all'][e], 'old': curves['old'][e], 'new': curves['new'][e]}})
                    + '\n')
    with open(os.path.join(d, 'epochs_val.jsonl'), 'w') as f:
        for e in range(n):
            f.write(json.dumps({'epoch': e + 1, 'val_acc': val_acc[e] if val_acc else None}) + '\n')
    return d
