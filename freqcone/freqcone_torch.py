"""FreqCone: frequency entailment cones for Generalized Category Discovery (PyTorch).

The repository's single FreqCone module. It descends from the mechanism study's module and is checked against
that study's JAX implementation by mechanism_study/parity_check.py (loss and gradients). Loss values are unchanged
(tests_freqcone/test_freqcone_module.py). Additions: aperture presets and construction
checks (APERTURES, cone_min_radius, aperture; FreqConeLoss raises ValueError on a degenerate cone),
the angle between b and u in the logs, and the two changes below:
  * ``pairwise_dist`` forms the Gram matrix instead of an N x M x n Mobius tensor (O(N M) memory;
    the reference form is kept as ``pairwise_dist_mobius``). At B = 128, n = 8192 the reference
    form needs about 0.5 GB per float32 intermediate, several times over, for each of the three
    distance matrices of L_con.
  * ``radial_lowpass`` takes ``out_size``: the band-limited view is resampled exactly by cropping
    its spectrum, which is alias-free whenever cutoff < out_size / 2.

Training-time losses for hyperbolic GCD code bases (here: Hyp-SelEx from HypCD). Inference is unchanged.

Integration sketch (inside the training step, after the base method has produced the
Poincare-ball points u1, u2 of the two augmented views with HypCD's mapping):

    from freqcone.freqcone_torch import FreqConeLoss, radial_lowpass

    # w_ent: choose on known-class validation; f_a, f_b: SNR 5 / 10 dB cut-offs from cutoffs.json
    freqcone = FreqConeLoss(c=0.1, w_ent=0.03, f_a=3.0, f_b=11.0, clip_r=2.0)
    z_a = proj_head(backbone(radial_lowpass(view1_images, freqcone.f_a)))   # Euclidean, (B, n)
    z_b = proj_head(backbone(radial_lowpass(view1_images, freqcone.f_b)))
    loss_fc, logs = freqcone(u1, u2, z_a, z_b)
    loss = loss_base + loss_fc

Design (v2), in one line each:
  * coarse views sit at fixed depths r_a < r_b (scaled norms); the network only chooses their direction
  * L_dir: coarse views learn to point at the other full view u2, which receives no gradient from it
  * L_ent: u1 must lie in the cone of z_b's point, which must lie in the cone of z_a's point; the full
    view's norm is detached so the cone can steer its direction but never shrink its depth
  * L_con: pairwise distances must not grow when going from full to blurred views
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- frequency views


def _gaussian_kernel(kernel_size: int, sigma: float, device, dtype):
    ax = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g[:, None] * g[None, :]


def gaussian_blur(x: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
    """Depthwise Gaussian blur, reflect padding. x: (B, C, H, W)."""
    k = _gaussian_kernel(kernel_size, sigma, x.device, x.dtype)
    weight = k.expand(x.shape[1], 1, kernel_size, kernel_size)
    pad = kernel_size // 2
    return F.conv2d(F.pad(x, (pad, pad, pad, pad), mode="reflect"), weight, groups=x.shape[1])


def _crop_spectrum(spec: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    """Keep the out_h x out_w lowest DFT frequencies (unshifted FFT layout, even sizes)."""
    H, W = spec.shape[-2:]
    rows = torch.cat([torch.arange(0, out_h // 2), torch.arange(H - out_h // 2, H)]).to(spec.device)
    cols = torch.cat([torch.arange(0, out_w // 2), torch.arange(W - out_w // 2, W)]).to(spec.device)
    return spec.index_select(-2, rows).index_select(-1, cols)


def radial_lowpass(x: torch.Tensor, cutoff, pre_blur: bool = True, out_size=None) -> torch.Tensor:
    """Ideal radial low-pass filter L_f, cut-off in cycles / image.

    x: (B, C, H, W). cutoff: float, or tensor of shape (B,) for per-sample cut-offs.
    out_size: None (keep H x W) or an even int: return the band-limited image sampled on an
    out_size x out_size grid. The spectrum is cropped, so the result is exact (no aliasing)
    when cutoff < out_size / 2; the cut-off stays in cycles / image of the input.
    """
    if pre_blur:
        x = gaussian_blur(x)
    H, W = x.shape[-2:]
    fy = torch.fft.fftfreq(H, device=x.device) * H
    fx = torch.fft.fftfreq(W, device=x.device) * W
    rad = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    if torch.is_tensor(cutoff) and cutoff.dim() == 1:
        mask = (rad[None] <= cutoff.view(-1, 1, 1)).to(x.dtype)[:, None]
        max_cut = float(cutoff.max())
    else:
        mask = (rad <= float(cutoff)).to(x.dtype)[None, None]
        max_cut = float(cutoff)
    spec = torch.fft.fft2(x) * mask
    if out_size is None or (out_size == H and out_size == W):
        return torch.fft.ifft2(spec).real
    if out_size % 2 or out_size > min(H, W) or max_cut >= out_size / 2:
        raise ValueError(f"out_size={out_size} must be even, <= input size, and > 2 * cutoff ({max_cut})")
    small = _crop_spectrum(spec, out_size, out_size)
    return torch.fft.ifft2(small).real * (out_size * out_size / (H * W))


# --------------------------------------------------------------------------- Poincare ball


def _safe_norm(v: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    return torch.sqrt(torch.sum(v * v, dim=dim, keepdim=keepdim) + 1e-12)


def to_ball(z: torch.Tensor, c: float = 0.05, clip_r: float = 2.3) -> torch.Tensor:
    """HypCD mapping: feature clipping then exp map at the origin."""
    n = _safe_norm(z, keepdim=True)
    z = z * torch.clamp(clip_r / n, max=1.0)
    n = _safe_norm(z, keepdim=True)
    sc = math.sqrt(c)
    return torch.tanh(sc * n) * z / (sc * n)


def to_fixed_depth(z: torch.Tensor, r_scaled: float, c: float) -> torch.Tensor:
    """Point with direction z / ||z|| at scaled norm sqrt(c) * ||p|| = r_scaled."""
    return z / _safe_norm(z, keepdim=True) * (r_scaled / math.sqrt(c))


def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
    xy = (x * y).sum(-1, keepdim=True)
    x2 = (x * x).sum(-1, keepdim=True)
    y2 = (y * y).sum(-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    den = 1 + 2 * c * xy + c ** 2 * x2 * y2
    return num / den.clamp_min(1e-12)


def pairwise_dist_mobius(u: torch.Tensor, v: torch.Tensor, c: float) -> torch.Tensor:
    """Reference form (the mechanism study's): materialises an (N, M, n) Mobius tensor."""
    m = mobius_add(-u[:, None, :], v[None, :, :], c)
    arg = (math.sqrt(c) * _safe_norm(m)).clamp(max=1 - 1e-5)
    return 2.0 / math.sqrt(c) * torch.atanh(arg)


def pairwise_dist(u: torch.Tensor, v: torch.Tensor, c: float) -> torch.Tensor:
    """Poincare distances between all rows of u (N, n) and v (M, n) -> (N, M), O(N M) memory.

    Same value as pairwise_dist_mobius: ||(-u) (+)_c v||^2 = ||u - v||^2 / den with
    den = 1 - 2c<u, v> + c^2 ||u||^2 ||v||^2 (the reference clamps den at 1e-12 and adds 1e-12
    under the square root; both are kept). Gram matrix and norms in float64.
    """
    ud, vd = u.double(), v.double()
    uv = ud @ vd.t()
    u2 = (ud * ud).sum(-1, keepdim=True)
    v2 = (vd * vd).sum(-1, keepdim=True).t()
    sq = (u2 + v2 - 2 * uv).clamp_min(0.0)
    den = (1 - 2 * c * uv + c ** 2 * u2 * v2).clamp_min(1e-12)
    arg = (math.sqrt(c) * torch.sqrt(sq / den + 1e-12)).clamp(max=1 - 1e-5)
    return (2.0 / math.sqrt(c) * torch.atanh(arg)).to(u.dtype)


def depth(u: torch.Tensor, c: float) -> torch.Tensor:
    """delta_c(u) = d_c(0, u)."""
    arg = (math.sqrt(c) * _safe_norm(u)).clamp(max=1 - 1e-5)
    return 2.0 / math.sqrt(c) * torch.atanh(arg)


# Coarse-view radii (r_a, r_b), as scaled norms sqrt(c)||p||, of the three aperture configurations of the
# pilot. K stays 0.1, so every level shares one aperture function and Ganea's transitivity holds; a wider
# cone is obtained by placing the coarse views closer to the origin (which also makes them shallower).
APERTURES = {'A1': (0.120, 0.200), 'A2': (0.104, 0.136), 'A3': (0.100, 0.114)}


def cone_min_radius(K: float) -> float:
    """eps_K = 2K / (1 + sqrt(1 + 4K^2)): below this scaled radius K(1 - r^2)/r > 1 and the cone degenerates."""
    return 2 * K / (1 + math.sqrt(1 + 4 * K * K))


def aperture(K: float, r: float) -> float:
    """Half-aperture psi (radians) of the cone at scaled radius r = sqrt(c)||p|| (pi/2 below eps_K)."""
    return math.asin(min(1.0, K * (1 - r * r) / r))


def scaled_depth(r: float, c: float) -> float:
    """delta_c of a point with scaled radius r: (2 / sqrt(c)) artanh(r)."""
    return 2.0 / math.sqrt(c) * math.atanh(r)


def cone_energy(p: torch.Tensor, q: torch.Tensor, c: float, K: float = 0.1, eps: float = 1e-6) -> torch.Tensor:
    """Ganea et al. (2018) energy max(0, Xi(p, q) - psi(p)), curvature -c by scaling.

    Zero means q lies inside the entailment cone of p ("p is a parent of q").
    """
    x, y = math.sqrt(c) * p, math.sqrt(c) * q
    xy = (x * y).sum(-1)
    x2 = (x * x).sum(-1)
    y2 = (y * y).sum(-1)
    num = xy * (1 + x2) - x2 * (1 + y2)
    den = torch.sqrt(x2 + 1e-12) * _safe_norm(x - y) * torch.sqrt((1 + x2 * y2 - 2 * xy).clamp_min(1e-12))
    xi = torch.acos((num / den).clamp(-1 + eps, 1 - eps))
    psi = torch.asin((K * (1 - x2) / torch.sqrt(x2 + 1e-12)).clamp(0.0, 1 - eps))
    return F.relu(xi - psi)


# --------------------------------------------------------------------------- losses


def _cosine_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).t()


def _cross_view_nce(sim: torch.Tensor, tau: float) -> torch.Tensor:
    target = torch.arange(sim.shape[0], device=sim.device)
    return F.cross_entropy(sim / tau, target)


class FreqConeLoss(nn.Module):
    """L = w_dir * L_dir + w_ent * L_ent + w_con * L_con   (FreqCone v2; v3 with detach_child_norm=False).

    w_ent must be chosen on known-class validation data (the synthetic study selected 0.03 from
    {0, 0.03, 0.1, 0.3, 1}, hence the default); it trades fine-grained accuracy for hierarchy:
    w_ent = 1 activated the cone but cut All sharply wherever it was active (paper/tab_regimes.tex,
    paper/tab_v3.tex). Watch logs["fc_depth_full"]:
    the cone can only act when the sharp views are deeper than the coarse view b (Lemma 3), i.e. when
    fc_depth_full > self.depth_b = (2/sqrt(c)) artanh(r_b). That threshold depends on c and r_b
    (1.813 at c = 0.05, r_b = 0.2; 1.282 at c = 0.1, r_b = 0.2); never hard-code it.

    Construction checks (ValueError): 0 < f_a < f_b; eps_K < r_a < r_b, so both cones are defined
    (psi < 90 degrees); and r_b < tanh(sqrt(c) * clip_r) when clip_r is given, so the coarse views lie
    inside the region that the base mapping can reach.

    Args of forward:
        u1   (B, n) Poincare point of full view 1 (the view the coarse views are made from)
        u2   (B, n) Poincare point of full view 2 (another augmentation of the same images)
        z_a  (B, n) Euclidean head output of view 1 low-passed at f_a (coarsest)
        z_b  (B, n) Euclidean head output of view 1 low-passed at f_b (f_a < f_b)
    """

    def __init__(self, c: float = 0.05, K: float = 0.1, r_a: float = 0.12, r_b: float = 0.2,
                 tau_a: float = 0.5, tau_b: float = 0.3,
                 w_dir: float = 0.5, w_ent: float = 0.03, w_con: float = 0.05,
                 f_a: float = 3.0, f_b: float = 11.0, detach_child_norm: bool = True, clip_r=None):
        super().__init__()
        if not 0 < f_a < f_b:
            raise ValueError(f'need 0 < f_a < f_b, got f_a={f_a}, f_b={f_b}')
        eps = cone_min_radius(K)
        if not eps < r_a < r_b:
            raise ValueError(f'need eps_K < r_a < r_b (eps_K = {eps:.4f} for K = {K}), got r_a={r_a}, r_b={r_b}: '
                             f'below eps_K the half-aperture is clamped at 90 degrees and the cone degenerates')
        if clip_r is not None and not r_b < math.tanh(math.sqrt(c) * clip_r):
            raise ValueError(f'need r_b < tanh(sqrt(c) * clip_r) = {math.tanh(math.sqrt(c) * clip_r):.4f}, got r_b={r_b}')
        self.c, self.K, self.r_a, self.r_b = c, K, r_a, r_b
        self.psi_a, self.psi_b = aperture(K, r_a), aperture(K, r_b)
        self.depth_a, self.depth_b = scaled_depth(r_a, c), scaled_depth(r_b, c)
        self.tau_a, self.tau_b = tau_a, tau_b
        self.w_dir, self.w_ent, self.w_con = w_dir, w_ent, w_con
        self.f_a, self.f_b = f_a, f_b
        self.detach_child_norm = detach_child_norm   # True = v2; False = v3 (cone may also move the norm)

    def forward(self, u1, u2, z_a, z_b):
        c = self.c
        ua = to_fixed_depth(z_a, self.r_a, c)
        ub = to_fixed_depth(z_b, self.r_b, c)
        target = u2.detach()
        l_dir = _cross_view_nce(_cosine_matrix(ua, target), self.tau_a) + \
            _cross_view_nce(_cosine_matrix(ub, target), self.tau_b)
        if self.detach_child_norm:
            n1 = _safe_norm(u1, keepdim=True)
            child = n1.detach() * u1 / n1
        else:
            child = u1
        l_ent = (cone_energy(ua, ub, c, self.K) + cone_energy(ub, child, c, self.K)).mean()
        da, db, du = pairwise_dist(ua, ua, c), pairwise_dist(ub, ub, c), pairwise_dist(u1, u1, c)
        off = 1.0 - torch.eye(u1.shape[0], device=u1.device, dtype=u1.dtype)
        l_con = ((F.relu(da - db) + F.relu(db - du)) * off).sum() / off.sum()
        total = self.w_dir * l_dir + self.w_ent * l_ent + self.w_con * l_con
        with torch.no_grad():
            in_cone = (cone_energy(ub, u1, c, self.K) < 1e-6).float().mean()
            cos_bu = F.cosine_similarity(ub, u1, dim=-1).clamp(-1, 1)
            angle_bu = torch.rad2deg(torch.acos(cos_bu)).mean()
        logs = {"fc_dir": l_dir.detach(), "fc_ent": l_ent.detach(), "fc_con": l_con.detach(),
                "fc_in_cone": in_cone, "fc_depth_full": depth(u1, c).mean().detach(), "fc_angle_b_u": angle_bu}
        return total, logs


@torch.no_grad()
def attribute_parent(u: torch.Tensor, coarse_protos: torch.Tensor, c: float, K: float = 0.1) -> torch.Tensor:
    """Retail use: index of the known class whose coarse prototype best entails u.

    u: (N, n) full-view points; coarse_protos: (M, n) points at depth r_b, e.g.
    to_fixed_depth(mean of z_b over each known class, r_b, c).
    """
    e = cone_energy(coarse_protos[None, :, :].expand(u.shape[0], -1, -1),
                    u[:, None, :].expand(-1, coarse_protos.shape[0], -1), c, K)
    return e.argmin(dim=1)


# --------------------------------------------------------------------------- self-test


def _self_test():
    torch.manual_seed(0)
    c = 0.05
    # 1. closed form vs. direct geodesic angle
    p = to_ball(torch.randn(4096, 16) * 3, c, clip_r=8.0)
    q = to_ball(torch.randn(4096, 16) * 3, c, clip_r=8.0)
    v = mobius_add(-p, q, c)
    direct = torch.acos(((v * p).sum(-1) / (_safe_norm(v) * _safe_norm(p))).clamp(-1, 1))
    x, y = math.sqrt(c) * p, math.sqrt(c) * q
    xy, x2, y2 = (x * y).sum(-1), (x * x).sum(-1), (y * y).sum(-1)
    closed = torch.acos(((xy * (1 + x2) - x2 * (1 + y2)) /
                         (x2.sqrt() * _safe_norm(x - y) * torch.sqrt(1 + x2 * y2 - 2 * xy))).clamp(-1, 1))
    err = (direct - closed).abs().max().item()
    # 2. gradients are finite; u2 gets no gradient from FreqCone; the full view's norm gets none from L_ent
    B, n = 64, 32
    z = torch.randn(4, B, n, requires_grad=True)
    u1, u2 = to_ball(z[0], c), to_ball(z[1], c)
    loss, logs = FreqConeLoss(c=c, w_con=0.0)(u1, u2, z[2], z[3])
    loss.backward()
    finite = torch.isfinite(z.grad).all().item()
    u2_untouched = z.grad[1].abs().max().item() == 0.0
    radial = (z.grad[0] * F.normalize(z[0], dim=-1)).sum(-1).abs().max().item()
    # 3. low-pass is monotone in the cut-off
    img = torch.randn(2, 3, 32, 32)
    d = [radial_lowpass(img[:1] - img[1:], f, pre_blur=False).norm().item() for f in range(0, 23)]
    mono = all(d[i] <= d[i + 1] + 1e-5 for i in range(len(d) - 1))
    print(f"closed-form vs geodesic angle, max abs err: {err:.2e}")
    print(f"loss {loss.item():.4f}, finite grads: {finite}, u2 receives no gradient: {u2_untouched}")
    print(f"max radial gradient on the full view from L_dir + L_ent: {radial:.2e}")
    print(f"low-pass distance monotone in cut-off: {mono}")
    assert err < 1e-4 and finite and u2_untouched and radial < 1e-5 and mono


if __name__ == "__main__":
    _self_test()
