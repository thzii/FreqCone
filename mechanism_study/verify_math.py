"""Numerical checks for the FreqCone propositions (numpy only).

Checks
  1. Ganea et al. (2018) closed form of Xi(x, y) equals the angle between the
     geodesic direction (-x (+) y) and the outward radial axis at x.
  2. Curvature scaling: Xi_c(p, q) = Xi_1(sqrt(c) p, sqrt(c) q).
  3. Lemma 3: zero cone energy  =>  depth(q) >= depth(p).
  4. Proposition 5: zero cone energy  =>  angle at the origin between p and q < psi(p).
  5. Proposition 2: slicing coordinates after the full-vector exp map shrinks the
     slice norm (strictly unless the other block is zero).
  6. Proposition 1: ||L_f (x - y)|| is non-decreasing in the cut-off f.
  7. Failure mode of a norm-detached cone: if the sharp view is shallower than its parent, the
     angle minimising Xi is far from 0 (the cone pushes the sharp view away from its parent).
  8. Moving the sharp view outward along its ray never increases Xi (so v3 cannot collapse it inward
     through L_ent).
"""
import numpy as np

rng = np.random.default_rng(0)


def mobius_add(x, y, c=1.0):
    xy = np.sum(x * y, -1, keepdims=True)
    x2 = np.sum(x * x, -1, keepdims=True)
    y2 = np.sum(y * y, -1, keepdims=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    den = 1 + 2 * c * xy + c ** 2 * x2 * y2
    return num / den


def depth(p, c=1.0):
    n = np.linalg.norm(p, axis=-1)
    return 2 / np.sqrt(c) * np.arctanh(np.sqrt(c) * n)


def xi_ganea(x, y):
    """Ganea et al. 2018, Eq. (28), unit ball (c = 1)."""
    xy = np.sum(x * y, -1)
    x2 = np.sum(x * x, -1)
    y2 = np.sum(y * y, -1)
    num = xy * (1 + x2) - x2 * (1 + y2)
    den = np.sqrt(x2) * np.linalg.norm(x - y, axis=-1) * np.sqrt(1 + x2 * y2 - 2 * xy)
    return np.arccos(np.clip(num / den, -1, 1))


def xi_c(p, q, c):
    return xi_ganea(np.sqrt(c) * p, np.sqrt(c) * q)


def xi_direct(x, y, c=1.0):
    """Angle between the initial direction of the geodesic x->y and x/||x||."""
    v = mobius_add(-x, y, c)  # direction of log_x(y) (positive scalar multiple)
    cos = np.sum(v * x, -1) / (np.linalg.norm(v, axis=-1) * np.linalg.norm(x, axis=-1))
    return np.arccos(np.clip(cos, -1, 1))


def psi_c(p, c, K=0.1):
    n = np.sqrt(c) * np.linalg.norm(p, axis=-1)
    return np.arcsin(np.clip(K * (1 - n ** 2) / n, -1, 1))


def exp_at(p, v, c=1.0):
    """exp_p(v) in the Poincare ball of curvature -c."""
    lam = 2 / (1 - c * np.sum(p * p, -1, keepdims=True))
    nv = np.linalg.norm(v, axis=-1, keepdims=True)
    second = np.tanh(np.sqrt(c) * lam * nv / 2) * v / (np.sqrt(c) * nv)
    return mobius_add(p, second, c)


def random_ball(n, d, rmin, rmax, c=1.0):
    u = rng.normal(size=(n, d))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    r = rng.uniform(rmin, rmax, size=(n, 1)) / np.sqrt(c)
    return u * r


def main():
    d, n = 16, 20000
    report = {}

    # 1. closed form vs direct geodesic angle (c = 1)
    x = random_ball(n, d, 0.1, 0.95)
    y = random_ball(n, d, 0.0, 0.95)
    err = np.abs(xi_ganea(x, y) - xi_direct(x, y))
    report["1_xi_closed_form_max_abs_err_rad"] = float(err.max())

    # 2. curvature scaling
    c = 0.05
    p = random_ball(n, d, 0.1, 0.95, c)
    q = random_ball(n, d, 0.0, 0.95, c)
    err2 = np.abs(xi_c(p, q, c) - xi_direct(p, q, c))
    report["2_xi_curvature_scaling_max_abs_err_rad"] = float(err2.max())

    # 3 + 4. points inside the cone of p: sample a direction within psi(p) of the
    # radial axis and walk along the geodesic for a random length
    p = random_ball(n, d, 0.12, 0.9, c)
    psi = psi_c(p, c)
    axis = p / np.linalg.norm(p, axis=1, keepdims=True)
    rand = rng.normal(size=(n, d))
    rand -= np.sum(rand * axis, 1, keepdims=True) * axis
    rand /= np.linalg.norm(rand, axis=1, keepdims=True)
    theta = rng.uniform(0, 1, size=n) * psi
    w = np.cos(theta)[:, None] * axis + np.sin(theta)[:, None] * rand
    length = rng.uniform(0.01, 3.0, size=(n, 1))
    q = exp_at(p, w * length, c)
    inside = xi_c(p, q, c) <= psi + 1e-6
    report["3a_fraction_sampled_points_inside_cone"] = float(inside.mean())
    gain = depth(q, c) - depth(p, c)
    report["3b_min_depth_gain_inside_cone"] = float(gain[inside].min())
    cos_o = np.sum(p * q, 1) / (np.linalg.norm(p, axis=1) * np.linalg.norm(q, axis=1))
    ang_o = np.arccos(np.clip(cos_o, -1, 1))
    report["4_max(angle_at_origin - psi)_inside_cone"] = float((ang_o - psi)[inside].max())

    # 3c. rejection sampling over the whole ball (independent check)
    p2 = random_ball(200000, 4, 0.12, 0.9, c)
    q2 = random_ball(200000, 4, 0.0, 0.99, c)
    m = xi_c(p2, q2, c) <= psi_c(p2, c)
    report["3c_rejection_samples_in_cone"] = int(m.sum())
    report["3c_min_depth_gain_rejection"] = float((depth(q2, c) - depth(p2, c))[m].min())

    # 5. slice coupling after the full-vector exp map
    za = rng.normal(size=(n, 8))
    zb = rng.normal(size=(n, 8)) * rng.uniform(0, 3, size=(n, 1))
    z = np.concatenate([za, zb], 1)
    def exp0(v):
        nv = np.linalg.norm(v, axis=1, keepdims=True)
        return np.tanh(np.sqrt(c) * nv) * v / (np.sqrt(c) * nv)
    sliced = np.linalg.norm(exp0(z)[:, :8], axis=1)
    alone = np.linalg.norm(exp0(za), axis=1)
    report["5_max(slice_norm - own_map_norm)"] = float((sliced - alone).max())
    report["5_mean_relative_shrink"] = float(np.mean(1 - sliced / alone))

    # 6. low-pass monotonicity on random 32x32 images
    H = W = 32
    fy = np.fft.fftfreq(H) * H
    fx = np.fft.fftfreq(W) * W
    rad = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    def lowpass(img, f):
        return np.real(np.fft.ifft2(np.fft.fft2(img) * (rad <= f)))
    worst = 0.0
    for _ in range(500):
        a, b = rng.normal(size=(2, H, W))
        vals = [np.linalg.norm(lowpass(a - b, f)) for f in np.arange(0, 23, 0.5)]
        worst = max(worst, float(np.max(-np.diff(vals))))
        # nesting L_f L_g = L_min(f,g)
    a = rng.normal(size=(H, W))
    nest = np.abs(lowpass(lowpass(a, 9), 4) - lowpass(a, 4)).max()
    report["6a_max_decrease_of_d_f_when_f_grows"] = worst
    report["6b_nesting_error_L9_L4_vs_L4"] = float(nest)

    # 7. failure mode found in the held-out runs: with the sharp view's norm held fixed, the angle
    #    theta* (at the origin) that minimises Xi(b, u) is 0 when u is deeper than b, but far from 0
    #    when u is shallower than b, so a norm-detached cone pushes u AWAY from its parent's direction.
    cc, rb = 0.05, 0.2                                     # curvature and scaled radius of the coarse view b
    bb = np.array([rb / np.sqrt(cc), 0.0])
    thetas = np.radians(np.linspace(0.0, 120.0, 2401))
    for du in (1.3, 4.6):                                  # sharp-view depths measured on held-out seeds
        ru = np.tanh(np.sqrt(cc) * du / 2) / np.sqrt(cc)
        uu = ru * np.stack([np.cos(thetas), np.sin(thetas)], 1)
        xs = xi_c(np.repeat(bb[None], len(thetas), 0), uu, cc)
        report[f"7_argmin_angle_deg_depth_u={du}_(depth_b=1.81)"] = float(np.degrees(thetas[np.argmin(xs)]))
        report[f"7_min_Xi_deg_depth_u={du}_(psi_b=28.7)"] = float(np.degrees(xs.min()))

    # 8. why removing that stop-gradient (v3) should not reopen the collapse toward the origin:
    #    moving the sharp view outward along its own ray never increases Xi(b, u)
    m8, d8 = 200000, 8
    b8 = np.zeros((m8, d8)); b8[:, 0] = rb / np.sqrt(cc)
    dirs = rng.normal(size=(m8, d8)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    r8 = np.tanh(np.sqrt(cc) * rng.uniform(0.3, 4.6, size=(m8, 1)) / 2) / np.sqrt(cc)
    grow = xi_c(b8, dirs * r8 * (1 + 1e-4), cc) - xi_c(b8, dirs * r8, cc)
    report["8_fraction_where_moving_outward_increases_Xi"] = float(np.mean(grow > 1e-12))

    for k, v in report.items():
        print(f"{k:48s} {v}")


if __name__ == "__main__":
    main()
