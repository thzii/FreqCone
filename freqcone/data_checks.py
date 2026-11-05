"""Identity and integrity of HypCD's data splits (known-class validation split, smoke subsets).

Supports the three fine-grained SSB datasets of the HypCD code: CUB (``data.cub.CustomCub2011``,
pandas ``data``), Stanford Cars (``data.stanford_cars.CarsDataset``, path list ``data``) and FGVC-Aircraft
(``data.fgvc_aircraft.FGVCAircraft``, ``samples``). An image is identified by its file (item_names), the same on
every machine and in every numbering: the list SHA-256 are computed on it. ``uq_idxs`` index the whole training set,
so the labelled, unlabelled and validation subsets can be compared directly. compact_uq_idxs renumbers the training
subsets and their copies for upstream's pseudo-label lookup; source_ids keeps giving the numbering of the whole
training set.
"""
import hashlib
import importlib
import os

import numpy as np

SUBSAMPLE_MODULES = {'cub': 'data.cub', 'scars': 'data.stanford_cars', 'aircraft': 'data.fgvc_aircraft'}


def file_tail(path):
    """File below the dataset root, independent of the machine: 'cars_train/00001.jpg'."""
    return '/'.join(str(path).replace('\\', '/').split('/')[-2:])


def item_names(ds):
    data = getattr(ds, 'data', None)
    if hasattr(data, 'columns') and 'filepath' in data.columns:            # CUB: path relative to the images folder
        return [str(p) for p in data['filepath']]
    if hasattr(ds, 'samples'):                                              # FGVC-Aircraft
        return [file_tail(p) for p, _ in ds.samples]
    if isinstance(data, list):                                              # Stanford Cars
        return [file_tail(p) for p in data]
    raise TypeError(f'unsupported dataset type {type(ds).__name__}')


def raw_targets(ds):
    """Class ids in the convention of the dataset's class splits (args.train_classes)."""
    data = getattr(ds, 'data', None)
    if hasattr(data, 'columns') and 'target' in data.columns:              # CUB, 1-based in the files
        return np.asarray(data['target']).astype(int) - 1
    if hasattr(ds, 'samples'):                                              # FGVC-Aircraft, 0-based
        return np.asarray([t for _, t in ds.samples]).astype(int)
    if hasattr(ds, 'target'):                                               # Stanford Cars, 1-based
        return np.asarray(ds.target).astype(int) - 1
    raise TypeError(f'unsupported dataset type {type(ds).__name__}')


def source_ids(ds):
    """uq_idxs in the numbering of the whole training set, also after compact_uq_idxs."""
    return getattr(ds, 'source_uq_idxs', ds.uq_idxs)


def parts(ds):
    """The datasets behind a merged training set, or the dataset itself."""
    return (ds.labelled_dataset, ds.unlabelled_dataset) if hasattr(ds, 'labelled_dataset') else (ds,)


def list_sha256(names):
    return hashlib.sha256('\n'.join(sorted(names)).encode('utf-8')).hexdigest()


def check_val_split(labelled, unlabelled, val, known_classes):
    """Raise AssertionError unless the validation split is non-empty, disjoint from both training subsets and
    known-class only. Returns its identity (SHA-256 of the sorted file list) and sizes."""
    v = {int(i) for i in source_ids(val)}
    lab = {int(i) for i in source_ids(labelled)}
    unl = {int(i) for i in source_ids(unlabelled)}
    if not v:
        raise AssertionError('validation split is empty')
    if v & lab:
        raise AssertionError(f'{len(v & lab)} validation images are also labelled training images')
    if v & unl:
        raise AssertionError(f'{len(v & unl)} validation images are also unlabelled training images')
    classes = set(raw_targets(val).tolist())
    novel = classes - {int(k) for k in known_classes}
    if novel:
        raise AssertionError(f'validation split contains {len(novel)} novel classes')
    return {'val_list_sha256': list_sha256(item_names(val)), 'n_val': len(v), 'n_val_classes': len(classes),
            'n_labelled': len(lab), 'n_unlabelled': len(unl)}


def check_merged_train(merged, known_classes):
    """Raise AssertionError unless the merged training set can drive SS-k-means and the supervised loss: labelled
    images exist, all of known classes, every known class among them. MergedDataset marks an image as labelled by
    its position, so these counts are the ones mask_lab gives. Returns the sizes recorded in run.json."""
    labelled, unlabelled = merged.labelled_dataset, merged.unlabelled_dataset
    known = sorted(int(k) for k in known_classes)
    if len(labelled) == 0:
        raise AssertionError('the merged training set has no labelled image')
    targets = raw_targets(labelled)
    novel = set(targets.tolist()) - set(known)
    if novel:
        raise AssertionError(f'{len(novel)} novel classes among the labelled images')
    per_class = np.bincount(targets, minlength=known[-1] + 1)[known]
    if (per_class == 0).any():
        raise AssertionError(f'{int((per_class == 0).sum())} known classes have no labelled image in the merged '
                             f'training set')
    return {'n_labelled': len(labelled), 'n_unlabelled': len(unlabelled),
            'labelled_per_known_min': int(per_class.min()), 'labelled_per_known_max': int(per_class.max())}


def compact_uq_idxs(labelled, unlabelled, copies=()):
    """Upstream looks pseudo-labels up with preds[np.argsort(uq_index)[uq_idxs]] (train(), three lines), which holds
    only if the merged training set's uq_idxs are exactly 0..N-1. Holding images out (--split_train_val,
    --smoke_frac) leaves gaps: ids go out of bounds or reach another image. Replace them by their rank, which
    changes nothing when there is no gap, here and in every copy of these images the loaders read (merged datasets
    or plain ones), so that one image has one number everywhere. The original ids stay in source_uq_idxs. Returns the
    number of gaps."""
    ids = np.concatenate([np.asarray(source_ids(labelled)), np.asarray(source_ids(unlabelled))]).astype(np.int64)
    if np.unique(ids).size != ids.size:
        raise AssertionError('duplicate uq_idxs in the merged training set')
    gaps = int(ids.max()) + 1 - ids.size
    if gaps == 0:
        return 0
    ordered = np.sort(ids)
    subsets = [labelled, unlabelled] + [part for copy in copies for part in parts(copy)]
    for ds in subsets:
        if not np.isin(source_ids(ds), ordered).all():
            raise AssertionError('a copy of the training set holds images that are not in the merged training set')
    for ds in subsets:
        ds.source_uq_idxs = source_ids(ds)
        ds.uq_idxs = np.searchsorted(ordered, ds.source_uq_idxs)
    return gaps


def check_uq_lookup(merged):
    """Raise AssertionError unless upstream's lookup can reach the right image: uq_idxs exactly 0..N-1."""
    ids = np.concatenate([np.asarray(part.uq_idxs) for part in parts(merged)]).astype(np.int64)
    if not np.array_equal(np.sort(ids), np.arange(ids.size)):
        raise AssertionError(f'uq_idxs of the merged training set are not 0..{ids.size - 1}: the pseudo-label lookup '
                             f'of upstream would reach the wrong images')


def uq_map_sha256(merged):
    """Identity of the renumbering: SHA-256 of the sorted 'old,new' pairs."""
    return list_sha256([f'{int(old)},{int(new)}' for part in parts(merged)
                        for old, new in zip(source_ids(part), part.uq_idxs)])


def keep_every(ds, k, dataset_name):
    """Smoke tests only: keep every k-th image of each class, the first one included, with the dataset module's own
    subsample_dataset (no random draws). Every class of the split stays represented."""
    subsample = importlib.import_module(SUBSAMPLE_MODULES[dataset_name]).subsample_dataset
    targets = raw_targets(ds)
    keep = np.concatenate([np.nonzero(targets == c)[0][::int(k)] for c in np.unique(targets)])
    return subsample(ds, np.sort(keep))
