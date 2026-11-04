"""SNR cut-offs for FreqCone's coarse views, computed without labels.

Source of the definition: Rastegar, Ghadimi Atigh, Mettes, Asano, Snoek, "Fourier Self-Supervision for
Fine-Grained Generalized Category Discovery", ECCV 2026, arXiv 2608.08963 (read on 2026-11-04 from
arxiv.org/html/2608.08963):
  * Section 4 ("Extraction of Critical Frequencies"):
        SNR(f_c) = 10 log( sum_{f <= f_c} |X(f)|^2 / sum_{f > f_c} |X(f)|^2 ),
    X the Fourier transform of an image, f_c the cut-off; the threshold is computed per category and averaged.
  * Section 9.1, Table 7: per-dataset cut-offs at 5 / 10 / 15 / 20 / 25 dB, "the maximum cutoff frequency that
    results in a particular SNR", determined before training from training images. Values in PAPER_TABLE7.
The paper does not state the frequency metric (its filters in Section 3 are box-shaped, 1(|u| < f, |v| < f)),
the resolution, the normalisation or the handling of DC. The authors' code is not released.

Fixed here, before any run:
  * metric: radial |k| = sqrt(k_u^2 + k_v^2) on the integer DFT grid (k = fftfreq(n) * n), with "f <= f_c" as in
    the mask of FreqCone's L_f (freqcone_torch.radial_lowpass without its pre-blur), so that "x^a is the 5 dB view"
    holds for the filter that training applies;
  * |X|^2: power summed over colour channels and averaged over images, DC included;
  * images: the training split only (labelled + unlabelled, validation split excluded), through HypCD's test
    transform (resize 256, centre crop 224, ImageNet normalisation), i.e. the tensors the network receives;
  * pooling: one mean spectrum over all those images and no labels (the paper's per-category average needs a
    class label for every image; GCD has labels only for the labelled part);
  * selection: the smallest integer f_c with SNR(f_c) >= L dB (SNR is non-decreasing in f_c).
"""
import math

import numpy as np
import torch

# Table 7 of arXiv 2608.08963 (Section 9.1): cut-offs at 5 / 10 / 15 / 20 / 25 dB. Reference only; training uses
# the values that scripts/freqcone/compute_cutoffs.py computes.
PAPER_TABLE7 = {
    'cub': {5: 2, 10: 8, 15: 23, 20: 45, 25: 55},
    'aircraft': {5: 2, 10: 9, 15: 27, 20: 50, 25: 55},
    'scars': {5: 3, 10: 11, 15: 27, 20: 47, 25: 55},
}
LEVELS_DB = (5.0, 10.0)          # f_a, f_b
DEFINITION = ('SNR(f) = 10 log10(sum_{|k|<=f} P(k) / sum_{|k|>f} P(k)); P = channel-summed |FFT2(x)|^2 averaged over '
              'all training images (labelled + unlabelled, validation split excluded, no labels), HypCD test '
              'transform at 224 px; |k| radial on the integer DFT grid, as in FreqCone L_f; cut-off = smallest '
              'integer f with SNR(f) >= level')
CITATION = ('Rastegar et al., Fourier Self-Supervision for Fine-Grained Generalized Category Discovery, ECCV 2026, '
            'arXiv 2608.08963: SNR in Section 4, cut-offs in Section 9.1, Table 7')


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
