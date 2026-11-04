"""Building HypCD's data splits outside the training script (cut-offs, data report, tests).

build_args reproduces the split settings of the training runs (SSB class splits, 50 % of the known-class images
labelled, GCD's validation split held out). set_root points a dataset module at another directory (for example
synthetic data) without editing config.py.
"""
import argparse
import importlib
import os

from freqcone.hypcd import REPO_ROOT

ROOT_VARS = {'scars': ('data.stanford_cars', 'car_root'), 'cub': ('data.cub', 'cub_root'),
             'aircraft': ('data.fgvc_aircraft', 'aircraft_root')}


def build_args(name, split_train_val=True):
    """The class splits and split settings the training script uses (reads data/ssb_splits/*.pkl)."""
    from data.get_datasets import get_class_splits
    ns = argparse.Namespace(dataset_name=name, use_ssb_splits=True, prop_train_labels=0.5,
                            split_train_val=split_train_val, smoke_frac=0, image_size=224, interpolation=3,
                            crop_pct=0.875, transform='imagenet')
    cwd = os.getcwd()
    os.chdir(REPO_ROOT)                     # upstream reads the split files relative to the repository root
    try:
        return get_class_splits(ns)
    finally:
        os.chdir(cwd)


def set_root(name, root):
    module, var = ROOT_VARS[name]
    setattr(importlib.import_module(module), var, os.path.abspath(root))
