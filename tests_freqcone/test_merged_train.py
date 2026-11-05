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

from conftest import FakeCars, FakeKMeans, needs_cuda
from data.data_utils import MergedDataset
from freqcone import pilot_tools as pt
from freqcone.data_checks import (check_merged_train, check_uq_lookup, check_val_split, compact_uq_idxs, raw_targets,
                                  source_ids, uq_map_sha256)
from freqcone.data_report import same_numbering
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


@pytest.mark.parametrize('split, smoke_frac', [(True, 0), (True, 3), (False, 3), (False, 0)])
def test_every_loader_reads_the_same_numbers(tmp_path, monkeypatch, split, smoke_frac):
    """train_loader and merge_train_loader share one object; merge_train_loader_test and unlabelled_train_loader
    read copies made before the renumbering."""
    _, _, _, _, loaders, datasets = build_upstream(tmp_path, monkeypatch, CPU, 1, split=split, smoke_frac=smoke_frac)
    merged = loaders['merge_train_loader'].dataset
    assert loaders['train_loader'].dataset is merged
    ids, _, _ = loader_order(loaders['merge_train_loader'])
    ids_test, _, _ = loader_order(loaders['merge_train_loader_test'])
    assert np.array_equal(ids, ids_test)
    unlabelled = np.concatenate([b[2].numpy() for b in loaders['unlabelled_train_loader']])
    assert np.array_equal(ids[len(merged.labelled_dataset):], unlabelled)
    assert same_numbering(merged, loaders['merge_train_loader_test'].dataset, loaders['unlabelled_train_loader'].dataset)
    if split:                                                    # held-out images keep the original numbers
        assert not hasattr(datasets['val'], 'source_uq_idxs')
        assert not set(datasets['val'].uq_idxs.tolist()) & set(source_ids(merged.labelled_dataset).tolist())


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


def test_renumbering_refuses_a_copy_of_other_images():
    images = torch.zeros(6, 3, 8, 8)
    labelled, unlabelled = Split(images[:3], [0, 1, 2], [0, 2, 4]), Split(images[3:], [0, 1, 2], [6, 8, 10])
    with pytest.raises(AssertionError, match='not in the merged training set'):
        compact_uq_idxs(labelled, unlabelled, (Split(images[:1], [0], [5]),))
    assert labelled.uq_idxs.tolist() == [0, 2, 4]                # nothing half done
    assert compact_uq_idxs(labelled, unlabelled, (Split(images[:1], [0], [6]),)) == 5


def test_map_identity_follows_the_renumbering(tmp_path, monkeypatch):
    _, _, _, _, loaders, _ = build_upstream(tmp_path, monkeypatch, CPU, 1)
    merged = loaders['merge_train_loader'].dataset
    before = uq_map_sha256(merged)
    assert before == uq_map_sha256(loaders['merge_train_loader_test'].dataset)
    part = merged.labelled_dataset
    part.uq_idxs = part.uq_idxs[::-1].copy()
    assert uq_map_sha256(merged) != before


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


def test_check_data_stops_a_numbering_that_the_lookup_cannot_use(tmp_path, monkeypatch):
    T, _, _, _, loaders, _ = build_upstream(tmp_path, monkeypatch, CPU, 1)
    for key in ('merge_train_loader', 'merge_train_loader_test'):
        T.FC_RUN.check_data(loaders[key])
        check_uq_lookup(loaders[key].dataset)
    part = loaders['merge_train_loader_test'].dataset.labelled_dataset
    part.uq_idxs = part.source_uq_idxs
    with pytest.raises(AssertionError, match='pseudo-label lookup'):
        T.FC_RUN.check_data(loaders['merge_train_loader_test'])


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
    assert counts['uq_map_sha256'] == uq_map_sha256(merged)


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


def installed(tmp_path, monkeypatch, commit):
    """What pip leaves for `pip install git+URL@commit`: a .dist-info with direct_url.json (PEP 610). commit None: a
    PyPI install, which has no such file."""
    info = tmp_path / 'kmeans_pytorch-0.3.dist-info'
    info.mkdir()
    (info / 'METADATA').write_text('Metadata-Version: 2.1\nName: kmeans_pytorch\nVersion: 0.3\n')
    if commit:
        (info / 'direct_url.json').write_text(json.dumps({
            'url': 'https://github.com/subhadarship/kmeans_pytorch',
            'vcs_info': {'vcs': 'git', 'requested_revision': commit, 'commit_id': commit}}))
    monkeypatch.syspath_prepend(str(tmp_path))


def test_the_pinned_build_passes_the_environment_check(tmp_path, monkeypatch):
    import kmeans_pytorch
    installed(tmp_path, monkeypatch, pt.KMEANS_COMMIT)
    monkeypatch.setattr(kmeans_pytorch, 'kmeans', lambda X, num_clusters, tqdm_flag=True: None)
    assert pt.kmeans_commit() == pt.KMEANS_COMMIT and pt.check_environment() == []


@pytest.mark.parametrize('commit, has_flag, expected', [
    (pt.KMEANS_COMMIT, False, 'no tqdm_flag'),                   # a build at the right commit cannot lack it; both checks run
    ('0' * 40, True, 'not the pinned build'),
    (None, True, 'no recorded commit'),
    (None, False, 'no tqdm_flag'),
])
def test_other_builds_fail_the_environment_check(tmp_path, monkeypatch, commit, has_flag, expected):
    import kmeans_pytorch
    installed(tmp_path, monkeypatch, commit)
    monkeypatch.setattr(kmeans_pytorch, 'kmeans', (lambda X, num_clusters, tqdm_flag=True: None) if has_flag else
                        (lambda X, num_clusters: None))
    problems = pt.check_environment()
    assert any(expected in p for p in problems)
    assert pt.KMEANS_INSTALL in problems[-1]


@needs_cuda
def test_this_machine_has_the_pinned_build():
    """The CUDA tests run upstream's own k-means: they mean something only on the build the training uses."""
    assert pt.check_environment() == []
