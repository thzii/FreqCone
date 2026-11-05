"""Section 2.3: the validation split, through upstream's own split code (data/cub.py, data/get_datasets.py)."""
import numpy as np
import pytest

from conftest import cub_args
from freqcone.data_checks import check_val_split, item_ids, keep_every, list_sha256, raw_targets


def build(args):
    from data.get_datasets import get_datasets
    train, test, unl_test, datasets = get_datasets('cub', None, None, args)
    return train, datasets


def test_val_is_disjoint_known_only_and_a_fifth_of_the_labelled(fake_cub):
    args = cub_args()
    train, datasets = build(args)
    lab, unl, val = train.labelled_dataset, train.unlabelled_dataset, datasets['val']
    info = check_val_split(lab, unl, val, args.train_classes)
    assert info['n_val'] > 0 and info['n_val_classes'] == 100
    whole = set(range(200 * fake_cub.PER_CLASS_TRAIN))
    v, l, u = (set(int(i) for i in d.uq_idxs) for d in (val, lab, unl))
    assert v | l | u == whole and not (v & l) and not (v & u) and not (l & u)       # a partition of the train set
    assert set(raw_targets(val).tolist()) <= set(args.train_classes)
    # 20 % (rounded down) of each known class's labelled images; about 3 per class here, as on real CUB
    full_lab, _ = build(cub_args(split_train_val=False))
    counts_full = np.bincount(raw_targets(full_lab.labelled_dataset), minlength=100)[:100]
    counts_val = np.bincount(raw_targets(val), minlength=100)[:100]
    assert (counts_val == (0.2 * counts_full).astype(int)).all()
    # the unlabelled set is the same with or without the split: the split only removes labelled images
    assert set(int(i) for i in full_lab.unlabelled_dataset.uq_idxs) == u


def test_val_is_independent_of_the_training_seed(fake_cub):
    shas = []
    for seed in (1, 7, 12345):
        np.random.seed(seed)                     # set_random_seed(args.seed) runs before get_datasets upstream
        train, datasets = build(cub_args())
        shas.append(check_val_split(train.labelled_dataset, train.unlabelled_dataset, datasets['val'],
                                    range(100))['val_list_sha256'])
    assert len(set(shas)) == 1


def test_split_flag_leaves_numpy_state_unchanged(fake_cub):
    """--split_train_val calls the dataset function a second time; it reseeds numpy itself, so the global RNG
    ends in the same state as with a single call, and training draws the same numbers."""
    states = []
    for split in (False, True):
        np.random.seed(0)
        build(cub_args(split_train_val=split))
        states.append(np.random.rand(3))
    assert np.array_equal(states[0], states[1])


class Part:
    """Minimal CUB-layout subset: uq_idxs and a pandas `data` with 1-based targets."""

    def __init__(self, idx, classes):
        import pandas as pd
        self.uq_idxs = np.asarray(idx)
        self.data = pd.DataFrame({'filepath': [f'{c:03d}/{i}.jpg' for i, c in zip(idx, classes)],
                                  'target': np.asarray(classes) + 1})


def test_checks_catch_overlap_and_novel_classes(fake_cub):
    args = cub_args()
    train, datasets = build(args)
    with pytest.raises(AssertionError, match='labelled'):
        check_val_split(train.labelled_dataset, train.unlabelled_dataset, train.labelled_dataset, args.train_classes)
    with pytest.raises(AssertionError, match='unlabelled'):
        check_val_split(train.labelled_dataset, train.unlabelled_dataset, train.unlabelled_dataset, args.train_classes)
    lab, unl = Part([0, 1], [0, 1]), Part([2, 3], [0, 150])
    with pytest.raises(AssertionError, match='novel'):
        check_val_split(lab, unl, Part([4], [150]), range(100))
    with pytest.raises(AssertionError, match='empty'):
        check_val_split(lab, unl, Part([], []), range(100))
    assert check_val_split(lab, unl, Part([4, 5], [0, 1]), range(100))['n_val'] == 2


def test_ids_and_smoke_subsample(fake_cub):
    train, datasets = build(cub_args())
    ids = item_ids(datasets['val'])
    assert ids[0].split(':', 1)[1].endswith('.jpg') and list_sha256(ids) == list_sha256(list(reversed(ids)))
    per_class = np.bincount(raw_targets(datasets['val']))
    small = keep_every(datasets['val'], 2, 'cub')
    assert len(small) == sum((n + 1) // 2 for n in per_class)                # every 2nd image within each class
    assert set(raw_targets(small).tolist()) == set(np.nonzero(per_class)[0].tolist())


def test_smoke_flag_shrinks_every_split(fake_cub):
    train, datasets = build(cub_args(smoke_frac=10))
    full, full_sets = build(cub_args())
    assert len(train.labelled_dataset) < len(full.labelled_dataset) / 2
    assert len(datasets['val']) < len(full_sets['val'])
    assert len(datasets['test']) < len(full_sets['test'])
    # no class disappears: a smoke run keeps labelled anchors for every known class
    assert set(raw_targets(train.labelled_dataset).tolist()) == set(raw_targets(full.labelled_dataset).tolist())
