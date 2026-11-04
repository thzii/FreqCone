"""Test setup: repository root on sys.path, and a stub for xformers when it is not installed.

models/vision_transformer2.py imports ``xformers.ops.SwiGLU`` at module level. xformers is a GPU
package (requirements.txt pins it for the training machine); the CPU tests never build that
backbone, so a stub class is enough to import train/train_HypSelEx.py.
"""
import importlib.util
import os
import sys
import types

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

if importlib.util.find_spec('xformers') is None:
    xformers = types.ModuleType('xformers')
    ops = types.ModuleType('xformers.ops')
    ops.SwiGLU = type('SwiGLU', (torch.nn.Module,), {})
    xformers.ops = ops
    sys.modules['xformers'] = xformers
    sys.modules['xformers.ops'] = ops

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not CUDA, reason='upstream K_Means hard-codes CUDA (run on the GPU machine)')


class FakeCub(torch.utils.data.Dataset):
    """Stand-in for data.cub.CustomCub2011 with the same attributes (pandas `data`, `uq_idxs`), sorted by class as
    CUB's images.txt is. Images are smooth random 32 x 32 tensors, fixed per image id; transforms are ignored."""
    N_CLASSES, PER_CLASS_TRAIN, PER_CLASS_TEST = 200, 30, 4

    def __init__(self, root=None, train=True, transform=None, target_transform=None, loader=None, download=True):
        rows, img_id = [], 1
        for c in range(1, self.N_CLASSES + 1):
            for j in range(self.PER_CLASS_TRAIN + self.PER_CLASS_TEST):
                rows.append((img_id, f'{c:03d}.species/img_{img_id:05d}.jpg', c, int(j < self.PER_CLASS_TRAIN)))
                img_id += 1
        data = pd.DataFrame(rows, columns=['img_id', 'filepath', 'target', 'is_training_img'])
        self.data = data[data.is_training_img == (1 if train else 0)]
        self.transform, self.target_transform, self.train = transform, target_transform, train
        self.uq_idxs = np.array(range(len(self)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        from freqcone.freqcone_torch import gaussian_blur
        sample = self.data.iloc[idx]
        g = torch.Generator().manual_seed(int(sample.img_id))
        img = gaussian_blur(torch.randn(1, 3, 32, 32, generator=g), 5, 2.0)[0]
        target = int(sample.target) - 1
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, self.uq_idxs[idx]


@pytest.fixture
def fake_cub(monkeypatch):
    import data.cub
    monkeypatch.setattr(data.cub, 'CustomCub2011', FakeCub)
    return FakeCub


def cub_args(**overrides):
    import argparse
    ns = argparse.Namespace(dataset_name='cub', train_classes=list(range(100)), unlabeled_classes=list(range(100, 200)),
                            prop_train_labels=0.5, split_train_val=True, smoke_frac=0, image_size=224,
                            interpolation=3, crop_pct=0.875, transform='imagenet')
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


class FakeKMeans:
    """CPU stand-in for upstream K_Means (which hard-codes CUDA): anchored clusters = labelled class means,
    each unlabelled point goes to the nearest mean. Same constructor and fit_mix signature, same label order."""

    def __init__(self, k, **kwargs):
        self.k, self.kwargs = k, kwargs

    def fit_mix(self, u_feats, l_feats, l_targets):
        classes = torch.unique(l_targets)
        centers = torch.stack([l_feats[l_targets == c].mean(0) for c in classes])
        u_lab = torch.cdist(u_feats, centers).argmin(1)
        cid = {int(c): i for i, c in enumerate(classes)}
        l_lab = torch.tensor([cid[int(t)] for t in l_targets])
        self.labels_ = torch.cat([l_lab, u_lab])
        self.cluster_centers_ = centers
