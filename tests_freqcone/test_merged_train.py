"""The merged training set behind SS-k-means and the supervised loss, on the data path of the training script.

Regressions:
  * held-out images (--split_train_val, --smoke_frac) left gaps in uq_idxs, and upstream's lookup
    preds[np.argsort(uq_index)[uq_idxs]] then went out of bounds or reached another image;
  * a loader without labelled images reached SS-k-means.
Each lookup test has a control showing that the same check fails on the data before the fix.
"""
import json
import types

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from conftest import FakeCars, FakeKMeans
from data.data_utils import MergedDataset
from freqcone.data_checks import check_merged_train, check_val_split, compact_uq_idxs, raw_targets, source_ids
from freqcone.hypcd import FCRun
from freqcone.pilot_tools import check_run_dir
from helpers import Split, build_upstream, fc_args, merged_loader, run_train

CPU = torch.device('cpu')


def loader_order(loader):
    """(ids, labels, mask) in the order upstream's extract_labeled_protos reads them."""
    batches = list(loader)
    return (np.concatenate([b[2].numpy() for b in batches]).astype(float),
            np.concatenate([b[1].numpy() for b in batches]),
            np.concatenate([b[3].numpy().ravel() for b in batches]).astype(bool))


def lookup_hits(loaders):
    """Share of the images of one training epoch for which upstream's lookup lands on an image of the same class and
    of the same id; raises IndexError on out-of-bounds ids, as train() does."""
    ids, labels, _ = loader_order(loaders['merge_train_loader'])
    order, hits, n = np.argsort(ids), 0, 0
    for _, class_labels, uq_idxs, _ in loaders['train_loader']:
        position = order[uq_idxs]
        hits += int(((labels[position] == class_labels.numpy()) & (ids[position] == uq_idxs.numpy())).sum())
        n += len(uq_idxs)
    return hits / n


def test_fixture_keeps_labelled_and_validation_images_of_every_known_class(tmp_path, monkeypatch):
    _, args, _, _, loaders, datasets = build_upstream(tmp_path, monkeypatch, CPU, 1)
    merged = loaders['merge_train_loader'].dataset
    per_class = np.bincount(raw_targets(merged.labelled_dataset), minlength=FakeCars.N_KNOWN)
    assert per_class.min() >= 2
    assert np.bincount(raw_targets(datasets['val']), minlength=FakeCars.N_KNOWN).min() >= 1
    assert set(raw_targets(datasets['val']).tolist()) == set(args.train_classes)


@pytest.mark.parametrize('split, smoke_frac', [(True, 0), (True, 3), (False, 3), (False, 0)])
def test_pseudo_label_lookup_reaches_the_right_image(tmp_path, monkeypatch, split, smoke_frac):
    _, _, _, _, loaders, _ = build_upstream(tmp_path, monkeypatch, CPU, 1, split=split, smoke_frac=smoke_frac)
    assert lookup_hits(loaders) == 1.0
    ids, _, _ = loader_order(loaders['merge_train_loader'])
    assert sorted(ids) == list(range(len(ids)))


@pytest.mark.parametrize('split, smoke_frac', [(True, 0), (True, 3), (False, 3)])
def test_lookup_fails_before_the_renumbering(tmp_path, monkeypatch, split, smoke_frac):
    """Control: on the ids the dataset code produces, the check above fails."""
    _, _, _, _, loaders, _ = build_upstream(tmp_path, monkeypatch, CPU, 1, split=split, smoke_frac=smoke_frac)
    for key in ('train_loader', 'merge_train_loader'):
        merged = loaders[key].dataset
        for part in (merged.labelled_dataset, merged.unlabelled_dataset):
            part.uq_idxs = source_ids(part)
    try:
        assert lookup_hits(loaders) < 0.5
    except IndexError:
        pass


def test_renumbering_changes_nothing_without_gaps(tmp_path, monkeypatch):
    _, _, _, _, loaders, _ = build_upstream(tmp_path, monkeypatch, CPU, 1, split=False)
    merged = loaders['merge_train_loader'].dataset
    assert not hasattr(merged.labelled_dataset, 'source_uq_idxs')
    before = merged.labelled_dataset.uq_idxs.copy(), merged.unlabelled_dataset.uq_idxs.copy()
    assert compact_uq_idxs(merged.labelled_dataset, merged.unlabelled_dataset) == 0
    assert np.array_equal(merged.labelled_dataset.uq_idxs, before[0])
    assert np.array_equal(merged.unlabelled_dataset.uq_idxs, before[1])


def test_renumbering_keeps_the_validation_checks_on_the_original_ids(tmp_path, monkeypatch):
    _, args, _, _, loaders, datasets = build_upstream(tmp_path, monkeypatch, CPU, 1)
    merged = loaders['merge_train_loader'].dataset
    assert hasattr(merged.labelled_dataset, 'source_uq_idxs')
    info = check_val_split(merged.labelled_dataset, merged.unlabelled_dataset, datasets['val'], args.train_classes)
    assert info['n_val'] == len(datasets['val'])
    used = set(source_ids(merged.labelled_dataset).tolist()) | set(source_ids(merged.unlabelled_dataset).tolist())
    assert not used & set(datasets['val'].uq_idxs.tolist())


def test_inert_without_flags():
    """Default flags: FCRun leaves the datasets alone."""
    images = torch.zeros(6, 3, 8, 8)
    merged = MergedDataset(Split(images[:3], [0, 1, 2], [0, 2, 4]), Split(images[3:], [0, 1, 2], [6, 8, 10]))
    run = FCRun(fc_args(), CPU, train_dataset=merged, datasets={})
    assert run.data_counts is None and merged.labelled_dataset.uq_idxs.tolist() == [0, 2, 4]


def merged_with(labelled_labels, n=12):
    images = torch.zeros(n, 3, 8, 8)
    k = len(labelled_labels)
    unlabelled = [i % 6 for i in range(n - k)]
    return merged_loader(images, torch.tensor(list(labelled_labels) + unlabelled), k)


def test_check_data_stops_a_run_without_labelled_anchors():
    run = FCRun(fc_args(), CPU)
    run.check_data(merged_with([0, 1, 2, 0]))
    with pytest.raises(AssertionError, match='no labelled image'):
        run.check_data(merged_with([]))
    with pytest.raises(AssertionError, match='1 known classes have no labelled image'):
        run.check_data(merged_with([0, 1, 0, 1]))
    with pytest.raises(AssertionError, match='novel classes among the labelled'):
        run.check_data(merged_with([0, 1, 2, 5]))


def test_check_data_refuses_a_loader_that_is_not_merged():
    """The defect of the first version of this fixture: a flat dataset whose mask is always 0."""
    class Flat(torch.utils.data.Dataset):
        labelled_dataset = None

        def __len__(self):
            return 4

        def __getitem__(self, i):
            return torch.zeros(3), 0, i, np.array([0])

    with pytest.raises(AssertionError, match='MergedDataset'):
        FCRun(fc_args(), CPU).check_data(DataLoader(Flat()))


def test_check_data_catches_a_leaked_validation_split(tmp_path, monkeypatch):
    T, args, _, _, loaders, datasets = build_upstream(tmp_path, monkeypatch, CPU, 1)
    T.FC_RUN.check_data(loaders['merge_train_loader'])
    T.FC_RUN.check_data(loaders['merge_train_loader_test'])
    novel = types.SimpleNamespace(uq_idxs=np.array([100000]), target=[8], data=['/x/n.jpg'])
    T.FC_RUN.val_dataset = novel
    with pytest.raises(AssertionError, match='novel classes'):
        T.FC_RUN.check_data(loaders['merge_train_loader'])
    leaked = source_ids(loaders['merge_train_loader'].dataset.labelled_dataset)[:2]
    T.FC_RUN.val_dataset = types.SimpleNamespace(uq_idxs=leaked, target=[1, 1], data=['/x/a.jpg', '/x/b.jpg'])
    with pytest.raises(AssertionError, match='labelled training images'):
        T.FC_RUN.check_data(loaders['merge_train_loader'])


def stub_upstream(T, monkeypatch):
    """CPU stand-ins for the three functions that hard-code CUDA; labels play the role of the pseudo-labels."""
    def fake_extract(m, loader, args):
        ids, labels, _ = loader_order(loader)
        protos = 0.01 * torch.randn(FakeCars.N_CLASSES, 16)
        return ids, torch.as_tensor(labels), [protos], [torch.as_tensor(labels)], dict(all_acc=.5, old_acc=.5, new_acc=.5)

    monkeypatch.setattr(T, 'extract_labeled_protos', fake_extract)
    monkeypatch.setattr(T, 'test_kmeans', lambda *a, **k: (0.5, 0.5, 0.5))
    monkeypatch.setattr(T, 'test', lambda m, l, a: {'all_acc': 0.5, 'old_acc': 0.6, 'new_acc': 0.4})
    T.FC_RUN.kmeans_cls = FakeKMeans


def test_one_epoch_trains_on_labelled_images_with_a_nonzero_supervised_loss(tmp_path, monkeypatch, capsys):
    T, args, model, head, loaders, datasets = build_upstream(tmp_path, monkeypatch, CPU, 1, eval_every_epoch=True)
    stub_upstream(T, monkeypatch)
    run_train(T, args, model, head, loaders)
    assert '[FreqCone] epoch 1: supervised SupCon on ' in capsys.readouterr().out
    assert check_run_dir(args.run_dir) == []
    with open(args.run_dir + '/epochs_val.jsonl') as f:
        sup = json.loads(f.readline())['sup_con']
    assert sup['n_labelled'] > 0 and sup['loss_distance'] > 0 and sup['loss_angle'] > 0
    with open(args.run_dir + '/run.json') as f:
        rec = json.load(f)
    assert rec['history'][0]['sup_con'] == sup
    merged = loaders['merge_train_loader'].dataset
    counts = rec['data_counts']
    expected = check_merged_train(merged, args.train_classes)
    assert {k: counts[k] for k in expected} == expected
    assert counts['n_val'] == len(datasets['val']) and counts['n_test_roi'] == len(datasets['test'])
    assert counts['uq_gaps_closed'] > 0 and counts['n_labelled'] + counts['n_unlabelled'] + counts['n_val'] == 192


def test_an_epoch_without_supervision_stops_the_run():
    run = FCRun(fc_args(), CPU)
    run._t_epoch = 0.0
    run.record_supcon(torch.zeros(4, dtype=torch.bool), 1.0, 1.0)
    with pytest.raises(RuntimeError, match='saw 0 labelled images'):
        run.end_epoch(0, 1.0, 1.0, 0.0)
    run.record_supcon(torch.ones(4, dtype=torch.bool), 0.0, 0.0)
    with pytest.raises(RuntimeError, match='saw 4 labelled images'):
        run.end_epoch(0, 1.0, 1.0, 0.0)
    run.record_supcon(torch.ones(4, dtype=torch.bool), 2.0, 3.0)
    run.record_supcon(torch.tensor([True, False]), 5.0, 3.0)
    run.end_epoch(0, 1.0, 1.0, 0.0)
    assert run.history[-1]['sup_con'] == {'n_labelled': 5, 'loss_distance': pytest.approx(13 / 5), 'loss_angle': 3.0}


def test_run_dir_check_requires_counts_and_supervision(tmp_path, monkeypatch):
    T, args, model, head, loaders, _ = build_upstream(tmp_path, monkeypatch, CPU, 1)
    stub_upstream(T, monkeypatch)
    run_train(T, args, model, head, loaders)
    assert check_run_dir(args.run_dir) == []
    path = args.run_dir + '/run.json'
    with open(path) as f:
        rec = json.load(f)
    good = json.dumps(rec)
    rec['data_counts'].pop('n_test_roi')
    rec['history'][0]['sup_con']['n_labelled'] = 0
    with open(path, 'w') as f:
        json.dump(rec, f)
    errs = check_run_dir(args.run_dir)
    assert 'missing data_counts.n_test_roi' in errs and any('supervised SupCon' in e for e in errs)
    with open(path, 'w') as f:
        f.write(good)


def test_kmeans_build_is_checked(monkeypatch):
    import kmeans_pytorch

    from freqcone.pilot_tools import check_environment
    monkeypatch.setattr(kmeans_pytorch, 'kmeans', lambda X, num_clusters, tqdm_flag=True: None)
    assert check_environment() == []
    monkeypatch.setattr(kmeans_pytorch, 'kmeans', lambda X, num_clusters: None)
    assert 'tqdm_flag' in check_environment()[0]
