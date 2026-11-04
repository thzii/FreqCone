"""SNR cut-offs: the source paper's formula applied to FreqCone's actual filter, from training images only."""
import importlib.util
import os

import numpy as np
import pytest
import torch

from conftest import ROOT, cub_args
from freqcone.cutoffs import PowerAccumulator, cutoffs_from_power, select_cutoff, snr_curve
from freqcone.freqcone_torch import gaussian_blur, radial_lowpass


def smooth_images(n, size=48, seed=0):
    g = torch.Generator().manual_seed(seed)
    return gaussian_blur(torch.randn(n, 3, size, size, generator=g, dtype=torch.float64), 7, 2.0)


def test_snr_is_the_energy_ratio_of_freqcone_lowpass():
    """SNR(f) from the mean spectrum = 10 log10(sum ||L_f x||^2 / sum ||x - L_f x||^2) with FreqCone's filter
    (no pre-blur): Parseval ties the definition to the filter that training applies."""
    x = smooth_images(6)
    acc = PowerAccumulator()
    acc.add(x[:4])
    acc.add(x[4:])
    curve = snr_curve(acc.mean())
    for f in (0, 1, 2, 5, 8, 13):
        low = radial_lowpass(x, float(f), pre_blur=False)
        direct = 10 * np.log10(float((low ** 2).sum()) / float(((x - low) ** 2).sum()))
        assert abs(curve[f] - direct) < 1e-8


def test_curve_is_monotone_and_selection_is_the_first_crossing():
    curve = snr_curve(_power(smooth_images(8)))
    finite = curve[np.isfinite(curve)]
    assert np.all(np.diff(finite) >= -1e-9) and np.isinf(curve[-1])
    for level in (5.0, 10.0):
        f = select_cutoff(curve, level)
        assert curve[f] >= level and (f == 0 or curve[f - 1] < level)
    levels, _ = cutoffs_from_power(_power(smooth_images(8)))
    assert levels[5.0] < levels[10.0]
    with pytest.raises(ValueError):
        select_cutoff(np.array([0.0, 1.0, 2.0]), 5.0)


def _power(x):
    acc = PowerAccumulator()
    acc.add(x)
    return acc.mean()


def _load_script():
    spec = importlib.util.spec_from_file_location('compute_cutoffs', os.path.join(ROOT, 'scripts', 'freqcone',
                                                                                  'compute_cutoffs.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_script_uses_training_images_only(fake_cub, monkeypatch):
    mod = _load_script()
    monkeypatch.setattr(mod, 'build_args', lambda name: cub_args())
    out = mod.compute('cub', 'cpu', batch_size=64, num_workers=0)
    from data.get_datasets import get_datasets
    train, _, _, datasets = get_datasets('cub', None, None, cub_args())
    n_train = len(train.labelled_dataset) + len(train.unlabelled_dataset)
    assert out['n_images'] == n_train and out['complete']
    assert out['validation_excluded']['n_val'] == len(datasets['val'])
    assert out['f_a'] < out['f_b'] and out['paper_table7'] == {5: 2, 10: 8, 15: 23, 20: 45, 25: 55}
    ids, _ = mod.cutoff_image_set(train, datasets['val'], range(100))
    assert not {int(i.split(':')[0]) for i in ids} & {int(i) for i in datasets['val'].uq_idxs}
