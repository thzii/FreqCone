"""Identity and integrity of HypCD's data splits (known-class validation split, smoke subsets).

Supports the three fine-grained SSB datasets of the HypCD code: CUB (``data.cub.CustomCub2011``,
pandas ``data``), Stanford Cars (``data.stanford_cars.CarsDataset``, path list ``data``) and FGVC-Aircraft
(``data.fgvc_aircraft.FGVCAircraft``, ``samples``). ``uq_idxs`` index the whole training set, so the
labelled, unlabelled and validation subsets can be compared directly.
"""
import hashlib
import importlib
import os

import numpy as np

SUBSAMPLE_MODULES = {'cub': 'data.cub', 'scars': 'data.stanford_cars', 'aircraft': 'data.fgvc_aircraft'}


def item_names(ds):
    data = getattr(ds, 'data', None)
    if hasattr(data, 'columns') and 'filepath' in data.columns:            # CUB: path relative to the images folder
        return [str(p) for p in data['filepath']]
    if hasattr(ds, 'samples'):                                              # FGVC-Aircraft
        return [os.path.basename(str(p)) for p, _ in ds.samples]
    if isinstance(data, list):                                              # Stanford Cars
        return [os.path.basename(str(p)) for p in data]
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


def item_ids(ds):
    return [f'{int(i)}:{n}' for i, n in zip(ds.uq_idxs, item_names(ds))]


def list_sha256(ids):
    return hashlib.sha256('\n'.join(sorted(ids)).encode('utf-8')).hexdigest()


def check_val_split(labelled, unlabelled, val, known_classes):
    """Raise AssertionError unless the validation split is non-empty, disjoint from both training subsets and
    known-class only. Returns its identity (SHA-256 of the sorted 'uq_idx:file' list) and sizes."""
    v = {int(i) for i in val.uq_idxs}
    lab = {int(i) for i in labelled.uq_idxs}
    unl = {int(i) for i in unlabelled.uq_idxs}
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
    return {'val_list_sha256': list_sha256(item_ids(val)), 'n_val': len(v), 'n_val_classes': len(classes),
            'n_labelled': len(lab), 'n_unlabelled': len(unl)}


def keep_every(ds, k, dataset_name):
    """Smoke tests only: keep positions 0, k, 2k, ... with the dataset module's own subsample_dataset."""
    subsample = importlib.import_module(SUBSAMPLE_MODULES[dataset_name]).subsample_dataset
    return subsample(ds, np.arange(0, len(ds), int(k)))
