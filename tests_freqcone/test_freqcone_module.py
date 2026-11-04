"""The vendored FreqCone module: memory-safe distances, exact resampling, gradient routing."""
import math

import torch
import torch.nn.functional as F

import freqcone.freqcone_torch as ft


def ball(n, d, seed, scale, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    return ft.to_ball(torch.randn(n, d, generator=g, dtype=dtype) * scale, c=0.05, clip_r=2.3)


def test_pairwise_dist_matches_mobius_form():
    for scale in (0.05, 0.3, 3.0):                      # shallow, mid and clipped (deep) points
        u, v = ball(30, 48, 0, scale), ball(25, 48, 1, scale)
        assert float((ft.pairwise_dist(u, v, 0.05) - ft.pairwise_dist_mobius(u, v, 0.05)).abs().max()) < 1e-9
    x = ball(20, 48, 2, 0.2)
    d = ft.pairwise_dist(x, x, 0.05)
    assert float(d.diagonal().abs().max()) < 1e-4 and torch.allclose(d, d.t(), atol=1e-12)


def _loss(z, detach_child_norm):
    loss = ft.FreqConeLoss(c=0.05, w_ent=1.0, w_con=0.05, detach_child_norm=detach_child_norm)
    u1, u2 = ft.to_ball(z[0], 0.05, 2.3), ft.to_ball(z[1], 0.05, 2.3)
    return loss(u1, u2, z[2], z[3])


def test_loss_and_gradients_equal_reference_form(monkeypatch):
    for scale in (0.05, 0.25, 2.0):
        for v3 in (False, True):
            z0 = torch.randn(4, 32, 24, generator=torch.Generator().manual_seed(3), dtype=torch.float64) * scale
            out = []
            for dist in (ft.pairwise_dist, ft.pairwise_dist_mobius):
                monkeypatch.setattr(ft, 'pairwise_dist', dist)
                z = z0.clone().requires_grad_(True)
                total, logs = _loss(z, detach_child_norm=not v3)
                total.backward()
                out.append((float(total.detach()), float(logs['fc_con']), z.grad.clone()))
            monkeypatch.undo()
            (t0, c0, g0), (t1, c1, g1) = out
            assert abs(t0 - t1) < 1e-9 and abs(c0 - c1) < 1e-9
            assert float((g0 - g1).abs().max()) < 1e-8


def test_gradient_routing():
    """u2 never receives gradient; with v2 and w_con = 0 the sharp view's norm receives none either."""
    z = (torch.randn(4, 64, 32, generator=torch.Generator().manual_seed(4)) * 0.3).requires_grad_(True)
    u1, u2 = ft.to_ball(z[0], 0.05, 2.3), ft.to_ball(z[1], 0.05, 2.3)
    loss, logs = ft.FreqConeLoss(c=0.05, w_ent=1.0, w_con=0.0)(u1, u2, z[2], z[3])
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(z.grad).all()
    assert float(z.grad[1].abs().max()) == 0.0
    radial = (z.grad[0] * F.normalize(z[0], dim=-1)).sum(-1).abs().max()
    assert float(radial) < 1e-6
    assert set(logs) == {'fc_dir', 'fc_ent', 'fc_con', 'fc_in_cone', 'fc_depth_full', 'fc_angle_b_u'}
    assert 0.0 <= float(logs['fc_angle_b_u']) <= 180.0


# ------------------------------------------------------------------ apertures (section 1.4)
TABLE = {'A1': (55.2, 28.7), 'A2': (72.0, 46.2), 'A3': (81.9, 60.0)}
DEPTHS = {0.1: {'A1': (0.763, 1.282), 'A2': (0.660, 0.866), 'A3': (0.635, 0.724)},
          0.05: {'A1': (1.079, 1.813), 'A2': (0.934, 1.224), 'A3': (0.897, 1.024)}}


def test_aperture_presets_match_the_table():
    assert abs(ft.cone_min_radius(0.1) - 0.0990) < 1e-4
    assert abs(ft.cone_min_radius(0.15) - 0.1468) < 1e-4 and abs(ft.cone_min_radius(0.2) - 0.1926) < 1e-4
    for name, (r_a, r_b) in ft.APERTURES.items():
        for c, clip_r in ((0.1, 1.5), (0.1, 2.0), (0.05, 1.5), (0.05, 2.3)):
            loss = ft.FreqConeLoss(c=c, K=0.1, r_a=r_a, r_b=r_b, clip_r=clip_r)      # all upstream (c, r) pairs
            assert abs(math.degrees(loss.psi_a) - TABLE[name][0]) < 0.2
            assert abs(math.degrees(loss.psi_b) - TABLE[name][1]) < 0.2
        for c, rows in DEPTHS.items():
            assert abs(ft.scaled_depth(r_a, c) - rows[name][0]) < 1e-3
            assert abs(ft.scaled_depth(r_b, c) - rows[name][1]) < 1e-3


def test_degenerate_or_unreachable_cones_are_rejected():
    import pytest
    with pytest.raises(ValueError, match='eps_K'):
        ft.FreqConeLoss(c=0.1, K=0.2, r_a=0.12, r_b=0.2)            # r_a < eps_K = 0.193: psi_a clamped at 90 deg
    with pytest.raises(ValueError, match='eps_K'):
        ft.FreqConeLoss(c=0.1, K=0.15, r_a=0.12, r_b=0.2)
    with pytest.raises(ValueError, match='eps_K'):
        ft.FreqConeLoss(c=0.1, K=0.1, r_a=0.2, r_b=0.12)            # order
    with pytest.raises(ValueError, match='tanh'):
        ft.FreqConeLoss(c=0.05, K=0.1, r_a=0.12, r_b=0.35, clip_r=1.5)   # tanh(sqrt(0.05) * 1.5) = 0.323
    with pytest.raises(ValueError, match='f_a'):
        ft.FreqConeLoss(f_a=8.0, f_b=8.0)
    assert ft.aperture(0.1, 0.05) == math.pi / 2                    # below eps_K the aperture is clamped


def test_default_w_ent_is_the_selected_lambda():
    assert ft.FreqConeLoss().w_ent == 0.03


def test_lowpass_resampling_is_exact():
    """A band-limited image resampled by spectrum cropping equals the full-size result sampled every 2 px."""
    x = torch.randn(2, 3, 224, 224, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    for f in (2.0, 8.0, 11.0):
        full = ft.radial_lowpass(x, f)
        small = ft.radial_lowpass(x, f, out_size=112)
        assert small.shape == (2, 3, 112, 112)
        assert float((small - full[..., ::2, ::2]).abs().max()) < 1e-10
    assert torch.equal(ft.radial_lowpass(x, 3.0, out_size=224), ft.radial_lowpass(x, 3.0))


def test_lowpass_rejects_aliasing_sizes():
    x = torch.randn(1, 3, 64, 64)
    for bad in (31, 128):
        try:
            ft.radial_lowpass(x, 5.0, out_size=bad)
        except ValueError:
            continue
        raise AssertionError(f'out_size={bad} accepted')
    try:
        ft.radial_lowpass(x, 20.0, out_size=32)        # 20 cycles cannot be represented on 32 px
    except ValueError:
        return
    raise AssertionError('cut-off above the new Nyquist limit accepted')


def test_coarse_depths():
    d = lambda r: 2 / math.sqrt(0.05) * math.atanh(r)
    assert abs(d(0.12) - 1.0785) < 1e-3 and abs(d(0.2) - 1.8123) < 1e-3
