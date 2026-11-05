"""Row (1) "Chuẩn GCD (theo code HypCD)" must equal upstream's own "Best Train Accuracies" line.

Both tests run upstream's train() from train/train_HypSelEx.py unchanged, on the data path of the training script
(helpers.build_upstream: upstream's dataset code, real MergedDataset, the script's sampler and loaders, validation
split held out), with the FreqCone hooks recording epochs_test.jsonl, then compare summarize_run's row (1) with what
upstream prints, to the precision of that line.
  * CPU: extract_labeled_protos / test_kmeans / test (which hard-code CUDA) are replaced by stand-ins returning
    scheduled accuracies, so the check covers the bookkeeping exactly: which triple upstream pairs with which epoch,
    its best-checkpoint rule (strict improvement of Old on the disjoint test split), and the last-epoch switch to
    test_kmeans on the unlabelled set.
  * CUDA (GPU machine): the real upstream functions, end to end, with and without the validation split and the
    label-order fix. It needs the GitHub build of kmeans_pytorch, as the training does (README section 2).
"""
import re

import numpy as np
import pytest
import torch

from conftest import FakeCars, FakeKMeans, needs_cuda
from freqcone.pilot_tools import check_run_dir
from freqcone.runs import read_jsonl
from freqcone.summarize_run import table
from helpers import build_upstream, run_train

BEST = re.compile(r'Best Train Accuracies: All ([\d.]+) \| Old ([\d.]+) \| New ([\d.]+)')
BEST_EPOCH = re.compile(r'Best Train Epochs:\s+(\d+)\s*$', re.M)


@pytest.mark.parametrize('disjoint_old, expect_epoch', [
    ([0.30, 0.45, 0.45, 0.40], 2),          # tie at epochs 2 and 3: upstream keeps the first
    ([0.30, 0.35, 0.40, 0.50], 4),          # best at the last epoch: test_kmeans on the unlabelled set
])
def test_row1_reproduces_best_train_accuracies(tmp_path, monkeypatch, capsys, disjoint_old, expect_epoch):
    epochs = 4
    T, args, model, head, loaders, _ = build_upstream(tmp_path, monkeypatch, torch.device('cpu'), epochs)
    T.FC_RUN.kmeans_cls = FakeKMeans
    ss_start = [(0.50, 0.60, 0.40), (0.55, 0.62, 0.51), (0.58, 0.66, 0.52), (0.60, 0.70, 0.55)]
    unlabelled_last = (0.6123, 0.7134, 0.5678)
    calls = {'extract': 0}

    def fake_extract(m, loader, args):
        e = calls['extract']
        calls['extract'] += 1
        batches = list(loader)
        ids = np.concatenate([b[2].numpy() for b in batches]).astype(float)         # as upstream builds them
        preds = torch.cat([b[1] for b in batches])
        g = torch.Generator().manual_seed(e)
        protos = 0.01 * torch.randn(FakeCars.N_CLASSES, 16, generator=g)            # close prototypes: small radii
        a, o, n = ss_start[e]
        return ids, preds, [protos], [preds], {'all_acc': a, 'old_acc': o, 'new_acc': n}

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
    tests = read_jsonl(args.run_dir + '/epochs_test.jsonl')
    row = next(r for r in table([], tests) if r['key'] == 'gcd')
    assert row['epoch'] == expect_epoch == printed_epoch + 1
    assert tuple(round(row[m], 4) for m in ('all', 'old', 'new')) == printed
    assert row['source'] == ('test_kmeans_unlabelled' if expect_epoch == epochs else 'ss_kmeans_epoch_start')
    assert tests[0]['gcd']['weights_after_epoch'] == 0                   # epoch 1 pairs with the pretrained weights
    assert check_run_dir(args.run_dir) == []


@needs_cuda
@pytest.mark.parametrize('split, fix', [(True, True), (True, False), (False, True), (False, False)])
def test_row1_matches_upstream_with_real_functions(tmp_path, monkeypatch, capsys, split, fix):
    T, args, model, head, loaders, _ = build_upstream(tmp_path, monkeypatch, torch.device('cuda'), 3, split=split,
                                                      fix_label_order=fix, eval_every_epoch=True)
    run_train(T, args, model, head, loaders)
    out = capsys.readouterr().out
    printed = tuple(float(v) for v in BEST.search(out).groups())
    tests = read_jsonl(args.run_dir + '/epochs_test.jsonl')
    row = next(r for r in table([], tests) if r['key'] == 'gcd')
    assert tuple(round(row[m], 4) for m in ('all', 'old', 'new')) == printed
    assert row['epoch'] == int(BEST_EPOCH.findall(out)[-1]) + 1
    assert check_run_dir(args.run_dir) == []
