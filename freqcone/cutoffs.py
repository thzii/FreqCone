"""SNR cut-offs for FreqCone's coarse views: FourEx's definition, computed without labels.

FourEx (Rastegar et al., "Fourier Self-Supervision for Fine-Grained Generalized Category Discovery",
ECCV 2026, arXiv 2608.08963) sets its filter thresholds from the spectral signal-to-noise ratio

    SNR(f_c) = 10 log10( sum_{f <= f_c} |X(f)|^2 / sum_{f > f_c} |X(f)|^2 )          (FX-P §4)

with per-dataset values in FX-A §9.1, Table 7. Transcription used here: PLANORA_SF_Method_Spec.md §4.3,
marked there as verbatim from the paper. FourEx's code is not released: github.com/SarahRastegar/FourEx
holds only a README ("Code is coming soon!", commit 02f5c37, checked 2026-11-04).

FourEx leaves the measurement open (spec, item U-09). Fixed here, before any run:
  * frequency metric: radial |k| = sqrt(k_u^2 + k_v^2) on the integer DFT grid (k = fftfreq(n) * n), with
    "f <= f_c" exactly as in the mask of FreqCone's L_f (freqcone_torch.radial_lowpass without pre-blur),
    so "x^a is the 5 dB view" holds for the filter that is actually applied;
  * |X|^2: power summed over colour channels, averaged over images, DC included;
  * images: the training split only (labelled + unlabelled, without the held-out validation images),
    through HypCD's test transform (resize 256, centre crop 224, ImageNet normalisation), i.e. the tensors
    the network receives;
  * pooling: one mean spectrum over all those images and no labels. FourEx instead averages per-category
    thresholds, which needs a class label for every image; GCD has labels only for the labelled part;
  * selection: the smallest integer f_c with SNR(f_c) >= L dB (SNR is non-decreasing in f_c).
"""
import math

import numpy as np
import torch

# FX-A Table 7 cut-offs at 5 / 10 / 15 / 20 / 25 dB, as transcribed in planora_sf (config.py, CUTOFF_TABLE).
# Reference only: the values used for training come from compute_cutoffs.py.
FOUREX_TABLE7 = {
    'cub': {5: 2, 10: 8, 15: 23, 20: 45, 25: 55},
    'aircraft': {5: 2, 10: 9, 15: 27, 20: 50, 25: 55},
    'scars': {5: 3, 10: 11, 15: 27, 20: 47, 25: 55},
}
LEVELS_DB = (5.0, 10.0)          # f_a, f_b
DEFINITION = ('SNR(f) = 10 log10(sum_{|k|<=f} P(k) / sum_{|k|>f} P(k)); P = channel-summed |FFT2(x)|^2 averaged over '
              'all training images (labelled + unlabelled, validation split excluded, no labels), HypCD test '
              'transform at 224 px; |k| radial on the integer DFT grid, as in FreqCone L_f; cut-off = smallest '
              'integer f with SNR(f) >= level')
CITATION = ('FourEx, Rastegar et al., ECCV 2026, arXiv 2608.08963: SNR formula FX-P §4; levels and per-dataset '
            'cut-offs FX-A §9.1, Table 7 (transcribed in PLANORA_SF_Method_Spec.md §4.3); official code not '
            'released (github.com/SarahRastegar/FourEx @ 02f5c37: README only)')


def radial_frequency(h: int, w: int, device=None) -> torch.Tensor:
    fy = torch.fft.fftfreq(h, device=device) * h
    fx = torch.fft.fftfreq(w, device=device) * w
    return torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)


class PowerAccumulator:
    """Running sum of the channel-summed power spectrum, so images can be streamed in batches."""

    def __init__(self):
        self.sum, self.n = None, 0

    def add(self, images: torch.Tensor):
        if images.ndim != 4:
            raise ValueError(f'expected (B, C, H, W), got {tuple(images.shape)}')
        power = torch.fft.fft2(images.double()).abs().pow(2).sum(dim=1).sum(dim=0)
        self.sum = power if self.sum is None else self.sum + power
        self.n += int(images.shape[0])

    def mean(self) -> torch.Tensor:
        if not self.n:
            raise ValueError('no images accumulated')
        return self.sum / self.n


def snr_curve(power: torch.Tensor, max_f=None) -> np.ndarray:
    """SNR(f) in dB for integer f = 0 .. max_f (default: the largest radius); +inf once nothing is above f."""
    rad = radial_frequency(*power.shape[-2:], device=power.device)
    total = float(power.sum())
    max_f = int(math.ceil(float(rad.max()))) if max_f is None else int(max_f)
    out = []
    for f in range(max_f + 1):
        signal = float(power[rad <= f].sum())
        noise = total - signal
        out.append(math.inf if noise <= 1e-12 * total else 10.0 * math.log10(signal / noise))
    return np.asarray(out)


def select_cutoff(curve: np.ndarray, level_db: float) -> int:
    hits = np.nonzero(curve >= level_db)[0]
    if hits.size == 0:
        raise ValueError(f'SNR never reaches {level_db} dB')
    return int(hits[0])


def cutoffs_from_power(power: torch.Tensor, levels=LEVELS_DB):
    curve = snr_curve(power)
    return {float(level): select_cutoff(curve, level) for level in levels}, curve
