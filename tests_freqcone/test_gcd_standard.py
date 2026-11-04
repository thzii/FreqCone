"""Row (1) "Chuẩn GCD (theo code HypCD)" must equal upstream's own "Best Train Accuracies" line.

Both tests run upstream's train() from train/train_HypSelEx.py unchanged, with the FreqCone hooks recording
epochs_test.jsonl, then compare summarize_run's row (1) with what upstream prints.
  * CPU: extract_labeled_protos / test_kmeans / test (which hard-code CUDA) are replaced by stand-ins returning
    scheduled accuracies, so the check covers the bookkeeping exactly: which triple upstream pairs with which epoch,
    its best-checkpoint rule (strict improvement of Old on the disjoint test split), and the last-epoch switch to
    test_kmeans on the unlabelled set.
  * CUDA (GPU machine): the real upstream functions, end to end.
"""
import importlib
import re
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from conftest import needs_cuda
from freqcone.hypcd import FCRun
from freqcone.pilot_tools import check_run_dir
from freqcone.runs import read_jsonl
from freqcone.summarize_run import table
from helpers import fc_args, make_model

N_KNOWN, N_NOVEL, N = 4, 4, 48
BEST = re.compile(r'Best Train Accuracies: All ([\d.]+) \| Old ([\d.]+) \| New ([\d.]+)')
BEST_EPOCH = re.compile(r'Best Train Epochs:\s+(\d+)\s*$', re.M)


class Train(Dataset):
    """Upstream MergedDataset items: ([view 1, view 2], label, uq_idx, labelled flag); labelled images first."""

    def __init__(self, x, y, lab):
        self.x, self.y, self.lab = x, y, lab

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        v = self.x[i]
        return [v, v.flip(-1)], int(self.y[i]), i, np.array([int(self.lab[i])])


class Single(Dataset):
    def __init__(self, x, y, with_mask=False):
        self.x, self.y, self.with_mask = x, y, with_mask

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return (self.x[i], int(self.y[i]), i) + ((np.array([0]),) if self.with_mask else ())


class Writer:
    def add_scalar(self, *a, **k):
        pass

    def add_scalars(self, *a, **k):
        pass


def build_upstream(tmp_path, monkeypatch, device, epochs):
    monkeypatch.setattr(sys, 'argv', ['pytest'])
    T = importlib.import_module('train.train_HypSelEx')
    g = torch.Generator().manual_seed(0)
    y = torch.arange(N) % (N_KNOWN + N_NOVEL)
    colours = torch.randn(N_KNOWN + N_NOVEL, 3, 1, 1, generator=g) * 3          # class-specific images
    x = colours[y] + 0.3 * torch.randn(N, 3, 32, 32, generator=g)
    lab = (y < N_KNOWN) & (torch.arange(N) % 2 == 0)
    order = torch.cat([torch.nonzero(lab).flatten(), torch.nonzero(~lab).flatten()])
    x, y, lab = x[order], y[order], lab[order]
    args = fc_args(run_dir=str(tmp_path / 'run'), epochs=epochs, num_labeled_classes=N_KNOWN,
                   num_unlabeled_classes=N_NOVEL, train_classes=range(N_KNOWN), batch_size=8, lr=0.01, momentum=0.9,
                   weight_decay=0.0, strategy='zero_one', cluster_momentum=1.0, unsupervised_smoothing=1.0,
                   train_report_interval=epochs, prototype_extraction_interval=1, distance='euclidean',
                   contrast_unlabel_only=False, sup_con_weight=0.35, n_views=2, temperature=1.0,
                   hyper_start_epoch=0, hyper_end_epoch=epochs, hyper_max_weight=1.0, writer=Writer(),
                   model_path=str(tmp_path / 'model.pt'), eval_funcs=['v2'], c=0.1, cr=2.0)
    model, head = make_model(args)
    model, head = model.to(device), head.to(device)
    li, ui = torch.nonzero(lab).flatten(), torch.nonzero(~lab).flatten()     # 12 labelled, 36 unlabelled
    mix = torch.cat([torch.cat([li[2 * b:2 * b + 2], ui[6 * b:6 * b + 6]]) for b in range(6)])
    loaders = dict(         # every batch mixes labelled and unlabelled images, as upstream's weighted sampler does
        train_loader=DataLoader(Train(x[mix], y[mix], lab[mix]), batch_size=8, shuffle=False, drop_last=True),
        test_loader=DataLoader(Single(x, y), batch_size=8),
        unlabelled_train_loader=DataLoader(Single(x[~lab], y[~lab]), batch_size=8),
        merge_train_loader=DataLoader(Train(x, y, lab), batch_size=8),
        merge_train_loader_test=DataLoader(Single(x, y, with_mask=True), batch_size=8))
    loaders['merge_train_loader_test'].dataset.labelled_dataset = Single(x[lab], y[lab])
    for name, value in (('args', args), ('device', device), ('FC_RUN', FCRun(args, device))):
        monkeypatch.setattr(T, name, value, raising=False)   # restored afterwards: other tests share the module
    return T, args, model, head, loaders, (x, y, lab)


def run_train(T, args, model, head, loaders):
    T.train(head, model, loaders['train_loader'], loaders['test_loader'], loaders['unlabelled_train_loader'],
            loaders['merge_train_loader'], loaders['merge_train_loader_test'], args)


@pytest.mark.parametrize('disjoint_old, expect_epoch', [
    ([0.30, 0.45, 0.45, 0.40], 2),          # tie at epochs 2 and 3: upstream keeps the first
    ([0.30, 0.35, 0.40, 0.50], 4),          # best at the last epoch: test_kmeans on the unlabelled set
])
def test_row1_reproduces_best_train_accuracies(tmp_path, monkeypatch, capsys, disjoint_old, expect_epoch):
    epochs = 4
    T, args, model, head, loaders, (x, y, lab) = build_upstream(tmp_path, monkeypatch, torch.device('cpu'), epochs)
    ss_start = [(0.50, 0.60, 0.40), (0.55, 0.62, 0.51), (0.58, 0.66, 0.52), (0.60, 0.70, 0.55)]
    unlabelled_last = (0.6123, 0.7134, 0.5678)
    calls = {'extract': 0}

    def fake_extract(m, loader, args):
        e = calls['extract']
        calls['extract'] += 1
        g = torch.Generator().manual_seed(e)
        preds = y.clone()
        protos = 0.01 * torch.randn(N_KNOWN + N_NOVEL, 16, generator=g)   # close prototypes: small radii
        a, o, n = ss_start[e]
        return np.arange(N, dtype=float), preds, [protos], [preds], {'all_acc': a, 'old_acc': o, 'new_acc': n}

    def fake_test_kmeans(m, loader, epoch, save_name, args, Use_GPU=True):
        if save_name == 'Test ACC':
            return 0.5, disjoint_old[epoch], 0.5
        return unlabelled_last

    monkeypatch.setattr(T, 'extract_labeled_protos', fake_extract)
    monkeypatch.setattr(T, 'test_kmeans', fake_test_kmeans)
    monkeypatch.setattr(T, 'test', lambda m, l, a: {'all_acc': 0.5, 'old_acc': 0.6, 'new_acc': 0.4})
    run_train(T, args, model, head, loaders)
    out = capsys.readouterr().out
    printed = tuple(float(v) for v in BEST.search(out).groups())
    printed_epoch = int(BEST_EPOCH.findall(out)[-1])
    tests = read_jsonl(str(tmp_path / 'run' / 'epochs_test.jsonl'))
    row = next(r for r in table([], tests) if r['key'] == 'gcd')
    assert row['epoch'] == expect_epoch == printed_epoch + 1
    assert tuple(round(row[m], 4) for m in ('all', 'old', 'new')) == printed
    assert row['source'] == ('test_kmeans_unlabelled' if expect_epoch == epochs else 'ss_kmeans_epoch_start')
    assert tests[0]['gcd']['weights_after_epoch'] == 0                   # epoch 1 pairs with the pretrained weights
    assert check_run_dir(str(tmp_path / 'run')) == []


@needs_cuda
def test_row1_matches_upstream_with_real_functions(tmp_path, monkeypatch, capsys):
    T, args, model, head, loaders, _ = build_upstream(tmp_path, monkeypatch, torch.device('cuda'), epochs=3)
    run_train(T, args, model, head, loaders)
    out = capsys.readouterr().out
    printed = tuple(float(v) for v in BEST.search(out).groups())
    row = next(r for r in table([], read_jsonl(str(tmp_path / 'run' / 'epochs_test.jsonl'))) if r['key'] == 'gcd')
    assert tuple(round(row[m], 4) for m in ('all', 'old', 'new')) == printed
