"""dist_matrix_lowmem gives upstream dist_matrix's values without the N x M x D tensor."""
import torch

from hyptorch.nn import ToPoincare
from hyptorch.pmath import dist_matrix, dist_matrix_lowmem


def ball(n, d, seed, c=0.05, r=2.3, dtype=torch.float64, scale=4.0):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, d, generator=g, dtype=dtype) * torch.rand(n, 1, generator=g, dtype=dtype) * scale
    return ToPoincare(c=c, clip_r=r, riemannian=False)(z)


def test_values_match_upstream_float64():
    for c, r in ((0.05, 2.3), (0.1, 2.0), (0.1, 1.5)):
        x, y = ball(40, 96, 0, c, r), ball(33, 96, 1, c, r)
        ref, new = dist_matrix(x, y, c=c), dist_matrix_lowmem(x, y, c=c)
        assert ref.shape == new.shape == (40, 33)
        assert float((ref - new).abs().max()) < 1e-9


def test_coordinate_slices_match():
    """SelEx's hierarchical SupCon applies the distance to coordinate prefixes of ball points.

    Self-distances differ on purpose: the 1e-12 floor under ||x - y||^2 gives d(x, x) ~ 2e-6 instead
    of ~0, which keeps sqrt's gradient finite. Every loss masks the diagonal (SupCon only uses it in
    the detached row maximum, a per-row constant shift).
    """
    x = ball(32, 256, 2)
    off = ~torch.eye(32, dtype=torch.bool)
    for k in (128, 64, 8):
        ref, new = dist_matrix(x[:, :k], x[:, :k], c=0.05), dist_matrix_lowmem(x[:, :k], x[:, :k], c=0.05)
        assert float((ref - new)[off].abs().max()) < 1e-9
        assert float(new.diagonal().abs().max()) < 1e-5


def test_float32_error_is_float32_sized():
    x32, y32 = ball(48, 512, 3, dtype=torch.float32), ball(48, 512, 4, dtype=torch.float32)
    exact = dist_matrix(x32.double(), y32.double(), c=0.05)
    err_up = (dist_matrix(x32, y32, c=0.05).double() - exact).abs().max()
    err_new = (dist_matrix_lowmem(x32, y32, c=0.05).double() - exact).abs().max()
    assert dist_matrix_lowmem(x32, y32, c=0.05).dtype == torch.float32
    assert float(err_new) < 1e-5 and float(err_new) <= 2 * float(err_up) + 1e-6


def test_gradients_match_and_diagonal_is_safe():
    """Self-distances (diagonal) are masked by the losses but still back-propagated: no NaN allowed."""
    base = ball(24, 64, 5)
    mask = ~torch.eye(24, dtype=torch.bool)
    w = torch.randn(24, 24, generator=torch.Generator().manual_seed(6), dtype=torch.float64)
    grads = []
    for fn in (dist_matrix, dist_matrix_lowmem):
        x = base.clone().requires_grad_(True)
        d = fn(x, x, c=0.05)
        (d[mask] * w[mask]).sum().backward()
        assert torch.isfinite(x.grad).all()
        grads.append(x.grad)
    assert float((grads[0] - grads[1]).abs().max() / grads[0].abs().max()) < 1e-8
