"""Mechanism-level experiment for FreqCone (JAX, CPU).

A synthetic, fully controlled GCD benchmark ("SpecHier-48") in which the parent
category lives in low spatial frequencies and the child category in high spatial
frequencies. It tests whether encoding the low-pass filtration as hyperbolic
entailment (FreqCone) helps novel-class discovery compared with: a spherical GCD
baseline, the HypCD hybrid loss, a simplified FourEx-style latent partition, and
naive low-pass positives.

Usage:  python toy_experiment.py --method freqcone --seed 0 --out results/
"""
import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

# --------------------------------------------------------------------------- data
S = 16                      # image side (pixels)
N_PARENT, N_CHILD = 8, 6    # 48 classes
N_KNOWN_CHILD = 3           # children 0..2 known, 3..5 novel, for every parent
PER_CLASS = 60


def torus_d2(size, my, mx):
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    dy = np.minimum(np.abs(yy - my), size - np.abs(yy - my))
    dx = np.minimum(np.abs(xx - mx), size - np.abs(xx - mx))
    return dy ** 2 + dx ** 2


def make_dataset(seed):
    rng = np.random.default_rng(10_000 + seed)
    yy, xx = np.meshgrid(np.arange(S), np.arange(S), indexing="ij")
    layouts = []
    for _ in range(N_PARENT):
        img = np.zeros((S, S))
        for _ in range(3):
            my, mx = rng.uniform(0, S, size=2)
            sig = rng.uniform(2.0, 3.0)
            amp = rng.uniform(0.6, 1.0) * rng.choice([-1.0, 1.0])
            img += amp * np.exp(-torus_d2(S, my, mx) / (2 * sig ** 2))
        img = (img - img.mean()) / img.std()
        layouts.append(img)
    thetas = np.deg2rad(np.arange(N_CHILD) * 30.0)       # shared texture pool
    perms = [rng.permutation(N_CHILD) for _ in range(N_PARENT)]
    X, Y, P, C = [], [], [], []
    for p in range(N_PARENT):
        envelope = 0.5 + 0.5 * np.abs(layouts[p]) / np.abs(layouts[p]).max()
        for c in range(N_CHILD):
            th = thetas[perms[p][c]]
            for _ in range(PER_CLASS):
                phase = rng.uniform(0, 2 * np.pi)
                tex = np.cos(2 * np.pi * 5.0 * (xx * np.cos(th) + yy * np.sin(th)) / S + phase)
                img = rng.uniform(0.8, 1.2) * layouts[p] + 0.55 * rng.uniform(0.8, 1.2) * tex * envelope
                img = np.roll(img, rng.integers(-3, 4, size=2), axis=(0, 1))
                img = rng.uniform(0.8, 1.2) * img + rng.normal(0, 0.2)
                img = img + rng.normal(0, 0.35, size=img.shape)
                X.append(img.astype(np.float32))
                Y.append(p * N_CHILD + c)
                P.append(p)
                C.append(c)
    X = np.stack(X)[..., None]
    Y, P, C = map(np.asarray, (Y, P, C))
    known = C < N_KNOWN_CHILD
    lab = np.zeros(len(Y), bool)
    for k in np.unique(Y[known]):
        idx = np.where(Y == k)[0]
        lab[rng.choice(idx, len(idx) // 2, replace=False)] = True
    # separate validation images of KNOWN classes only (never trained on), used to pick
    # hyper-parameters the way GCD papers do (no novel-class labels involved)
    Xv, Yv = [], []
    for p in range(N_PARENT):
        envelope = 0.5 + 0.5 * np.abs(layouts[p]) / np.abs(layouts[p]).max()
        for c in range(N_KNOWN_CHILD):
            th = thetas[perms[p][c]]
            for _ in range(20):
                phase = rng.uniform(0, 2 * np.pi)
                tex = np.cos(2 * np.pi * 5.0 * (xx * np.cos(th) + yy * np.sin(th)) / S + phase)
                img = rng.uniform(0.8, 1.2) * layouts[p] + 0.55 * rng.uniform(0.8, 1.2) * tex * envelope
                img = np.roll(img, rng.integers(-3, 4, size=2), axis=(0, 1))
                img = rng.uniform(0.8, 1.2) * img + rng.normal(0, 0.2)
                img = img + rng.normal(0, 0.35, size=img.shape)
                Xv.append(img.astype(np.float32))
                Yv.append(p * N_CHILD + c)
    return dict(X=X, Y=Y, P=P, known=known, lab=lab, Xval=np.stack(Xv)[..., None], Yval=np.asarray(Yv))


def augment(rng, x):
    out = np.empty_like(x)
    for i in range(len(x)):
        img = np.roll(x[i], rng.integers(-2, 3, size=2), axis=(0, 1))
        img = rng.uniform(0.85, 1.15) * img + rng.normal(0, 0.1)
        out[i] = img + rng.normal(0, 0.1, size=img.shape)
    return out


# --------------------------------------------------------------------------- model
def init_params(key, widths=(32, 64, 96, 128), proj_dim=32):
    params = {}
    cin = 1
    keys = jax.random.split(key, len(widths) + 2)
    for i, w in enumerate(widths):
        params[f"c{i}"] = jax.random.normal(keys[i], (3, 3, cin, w)) * jnp.sqrt(2.0 / (9 * cin))
        params[f"b{i}"] = jnp.zeros((w,))
        cin = w
    params["p1"] = jax.random.normal(keys[-2], (cin, 128)) * jnp.sqrt(2.0 / cin)
    params["q1"] = jnp.zeros((128,))
    params["p2"] = jax.random.normal(keys[-1], (128, proj_dim)) * jnp.sqrt(1.0 / 128)
    params["q2"] = jnp.zeros((proj_dim,))
    return params


def backbone(params, x):
    h = x
    for i, s in enumerate((1, 2, 2, 2)):
        h = jax.lax.conv_general_dilated(h, params[f"c{i}"], (s, s), "SAME",
                                         dimension_numbers=("NHWC", "HWIO", "NHWC"))
        h = jax.nn.gelu(h + params[f"b{i}"])
    return h.mean(axis=(1, 2))


def head(params, h):
    return jax.nn.gelu(h @ params["p1"] + params["q1"]) @ params["p2"] + params["q2"]


# --------------------------------------------------------------------------- frequency
_fy = np.fft.fftfreq(S) * S
RAD = jnp.asarray(np.sqrt(_fy[:, None] ** 2 + _fy[None, :] ** 2), jnp.float32)


def lowpass(x, f):
    """Ideal radial low-pass with cut-off f (cycles / image), x: (B,S,S,1)."""
    spec = jnp.fft.fft2(x[..., 0])
    return jnp.real(jnp.fft.ifft2(spec * (RAD <= f)))[..., None]


def lowpass_per_sample(x, f):
    spec = jnp.fft.fft2(x[..., 0])
    mask = (RAD[None] <= f[:, None, None])
    return jnp.real(jnp.fft.ifft2(spec * mask))[..., None]


# --------------------------------------------------------------------------- hyperbolic
C_CURV, CLIP_R, K_CONE = 0.05, 2.3, 0.1
EPS = 1e-6


def safe_norm(v, axis=-1, keepdims=False):
    return jnp.sqrt(jnp.sum(v * v, axis=axis, keepdims=keepdims) + 1e-12)


def to_ball(z, c=C_CURV, r=CLIP_R):
    n = safe_norm(z, keepdims=True)
    z = z * jnp.minimum(1.0, r / n)
    n = safe_norm(z, keepdims=True)
    return jnp.tanh(jnp.sqrt(c) * n) * z / (jnp.sqrt(c) * n)


def mobius_add(x, y, c=C_CURV):
    xy = jnp.sum(x * y, -1, keepdims=True)
    x2 = jnp.sum(x * x, -1, keepdims=True)
    y2 = jnp.sum(y * y, -1, keepdims=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    den = 1 + 2 * c * xy + c ** 2 * x2 * y2
    return num / (den + 1e-12)


def pdist(u, v, c=C_CURV):
    """Pairwise Poincare distances, u: (N,d), v: (M,d) -> (N,M)."""
    m = mobius_add(-u[:, None, :], v[None, :, :], c)
    arg = jnp.clip(jnp.sqrt(c) * safe_norm(m), 0.0, 1 - 1e-5)
    return 2.0 / jnp.sqrt(c) * jnp.arctanh(arg)


def cosine_matrix(u, v):
    un = u / safe_norm(u, keepdims=True)
    vn = v / safe_norm(v, keepdims=True)
    return un @ vn.T


def cone_energy(p, q, c=C_CURV, K=K_CONE):
    """Ganea et al. (2018) entailment energy, curvature -c via scaling."""
    x, y = jnp.sqrt(c) * p, jnp.sqrt(c) * q
    xy = jnp.sum(x * y, -1)
    x2 = jnp.sum(x * x, -1)
    y2 = jnp.sum(y * y, -1)
    num = xy * (1 + x2) - x2 * (1 + y2)
    den = jnp.sqrt(x2 + 1e-12) * safe_norm(x - y) * jnp.sqrt(jnp.maximum(1 + x2 * y2 - 2 * xy, 1e-12))
    xi = jnp.arccos(jnp.clip(num / den, -1 + EPS, 1 - EPS))
    psi = jnp.arcsin(jnp.clip(K * (1 - x2) / jnp.sqrt(x2 + 1e-12), 0.0, 1 - EPS))
    return jax.nn.relu(xi - psi)


# --------------------------------------------------------------------------- losses
def nt_xent(sim, tau):
    n = sim.shape[0]
    b = n // 2
    logits = sim / tau - 1e9 * jnp.eye(n)
    pos = (jnp.arange(n) + b) % n
    return jnp.mean(jax.nn.logsumexp(logits, axis=1) - logits[jnp.arange(n), pos])


def supcon(sim, y, labeled, tau):
    n = sim.shape[0]
    valid = labeled[:, None] & labeled[None, :] & ~jnp.eye(n, dtype=bool)
    logits = jnp.where(valid, sim / tau, -1e9)
    logp = logits - jax.nn.logsumexp(logits, axis=1, keepdims=True)
    pos = valid & (y[:, None] == y[None, :])
    npos = jnp.sum(pos, 1)
    per = -jnp.sum(jnp.where(pos, logp, 0.0), 1) / jnp.maximum(npos, 1)
    use = labeled & (npos > 0)
    return jnp.sum(jnp.where(use, per, 0.0)) / jnp.maximum(jnp.sum(use), 1)


def cross_view_nce(sim, tau):
    """Anchor i (rows) vs candidates j (cols); positive is j == i."""
    logits = sim / tau
    return jnp.mean(jax.nn.logsumexp(logits, axis=1) - jnp.diag(logits))


LAMBDA_SUP, TAU = 0.35, 0.1

# Design history (all variants are reported):
#   v0: plain losses, coarse depth learned, L_dir pulls the full view too
#   v1: + the full view's norm is detached inside L_ent
#   v2 (freqcone): + coarse views at fixed radii r_a < r_b, + stop-gradient on the full view in L_dir
_V2 = dict(fixed_radius=True, sg_dir_target=True, detach_child_norm=True, use_ent=True, use_con=True)
FREQCONE_VARIANTS = {
    "freqcone_v0": dict(fixed_radius=False, sg_dir_target=False, detach_child_norm=False, use_ent=True, use_con=True),
    "freqcone_v1": dict(fixed_radius=False, sg_dir_target=False, detach_child_norm=True, use_ent=True, use_con=True),
    "freqcone": _V2,
    "freqcone_no_con": dict(_V2, use_con=False),
    "freqcone_no_ent": dict(_V2, use_ent=False),
    # v3: v2 without the stop-gradient on the sharp-view norm in L_ent. Motivated by the held-out
    # diagnostics of v2: when the base model keeps sharp views shallower than the fixed coarse depth,
    # Lemma 3 makes cone membership impossible and a norm-detached cone can only rotate the sharp view
    # AWAY from its coarse view. Letting L_ent move the norm allows the sharp view to pass its parent.
    "freqcone_v3": dict(_V2, detach_child_norm=False),
}


def base_loss(u_all, y_all, lab_all, hyperbolic, alpha):
    if not hyperbolic:
        sim = cosine_matrix(u_all, u_all)
        return (1 - LAMBDA_SUP) * nt_xent(sim, TAU) + LAMBDA_SUP * supcon(sim, y_all, lab_all, TAU)
    s_ang = cosine_matrix(u_all, u_all)
    s_dis = -pdist(u_all, u_all)
    lu = alpha * nt_xent(s_dis, TAU) + (1 - alpha) * nt_xent(s_ang, TAU)
    ls = alpha * supcon(s_dis, y_all, lab_all, TAU) + (1 - alpha) * supcon(s_ang, y_all, lab_all, TAU)
    return (1 - LAMBDA_SUP) * lu + LAMBDA_SUP * ls


def make_loss(method, cfg):
    hyperbolic = method != "gcd"

    def embed(params, x):
        z = head(params, backbone(params, x))
        return to_ball(z) if hyperbolic else z

    def loss_fn(params, batch, alpha, key):
        x1, x2, y, lab = batch["x1"], batch["x2"], batch["y"], batch["lab"]
        bsz = x1.shape[0]
        u1, u2 = embed(params, x1), embed(params, x2)
        u_all = jnp.concatenate([u1, u2])
        y_all = jnp.concatenate([y, y])
        lab_all = jnp.concatenate([lab, lab])
        loss = base_loss(u_all, y_all, lab_all, hyperbolic, alpha)
        logs = {"base": loss}

        if method == "fourex_partition":
            k1, k2 = jax.random.split(key)
            fl = jax.random.choice(k1, jnp.array([1.0, 1.5, 2.0, 2.5, 3.0]), (bsz,))
            fh = jax.random.choice(k2, jnp.array([1.0, 1.5, 2.0, 2.5, 3.0]), (bsz,))
            ul = embed(params, lowpass_per_sample(x1, fl))
            uh = embed(params, x1 - lowpass_per_sample(x1, fh))
            half = u1.shape[1] // 2
            extra = 0.0
            for part, (sl, other) in {"low": (slice(0, half), ul), "high": (slice(half, None), uh)}.items():
                a = jnp.concatenate([u1[:, sl], other[:, sl]])
                s_ang = cosine_matrix(a, a)
                s_dis = -pdist(a, a)
                lu = alpha * nt_xent(s_dis, TAU) + (1 - alpha) * nt_xent(s_ang, TAU)
                ls = alpha * supcon(s_dis, y_all, lab_all, TAU) + (1 - alpha) * supcon(s_ang, y_all, lab_all, TAU)
                term = (1 - LAMBDA_SUP) * lu + LAMBDA_SUP * ls
                logs[part] = term
                extra = extra + cfg["w_fourex"] * term
            loss = loss + extra

        var = FREQCONE_VARIANTS.get(method)
        if method == "naive_lowpass":
            ua = embed(params, lowpass(x1, cfg["f_a"]))
            ub = embed(params, lowpass(x1, cfg["f_b"]))
        elif var is not None:
            if var["fixed_radius"]:
                # v2: the frequency level sets the depth; the network only chooses the direction
                za = head(params, backbone(params, lowpass(x1, cfg["f_a"])))
                zb = head(params, backbone(params, lowpass(x1, cfg["f_b"])))
                ua = za / safe_norm(za, keepdims=True) * (cfg["r_a"] / jnp.sqrt(C_CURV))
                ub = zb / safe_norm(zb, keepdims=True) * (cfg["r_b"] / jnp.sqrt(C_CURV))
            else:
                ua = embed(params, lowpass(x1, cfg["f_a"]))
                ub = embed(params, lowpass(x1, cfg["f_b"]))

        if method == "naive_lowpass":
            t = 0.0
            for uc in (ua, ub):
                s = alpha * cross_view_nce(-pdist(uc, u2), TAU) + (1 - alpha) * cross_view_nce(cosine_matrix(uc, u2), TAU)
                t = t + s
            logs["naive"] = t
            loss = loss + cfg["w_naive"] * t

        if var is not None:
            # v2: parents learn to point at their child; the child's direction is not pulled by L_dir
            target = jax.lax.stop_gradient(u2) if var["sg_dir_target"] else u2
            l_dir = cross_view_nce(cosine_matrix(ua, target), cfg["tau_a"]) + cross_view_nce(cosine_matrix(ub, target), cfg["tau_b"])
            loss = loss + cfg["w_dir"] * l_dir
            logs["dir"] = l_dir
            if var["use_ent"]:
                if var["detach_child_norm"]:
                    # v1+: the cone may steer the child's DIRECTION but not shrink its depth
                    n1 = safe_norm(u1, keepdims=True)
                    child = jax.lax.stop_gradient(n1) * u1 / n1
                else:
                    child = u1
                l_ent = jnp.mean(cone_energy(ua, ub) + cone_energy(ub, child))
                floor = jnp.mean(jax.nn.relu(cfg["eps_floor"] - jnp.sqrt(C_CURV) * safe_norm(ua))
                                 + jax.nn.relu(cfg["eps_floor"] - jnp.sqrt(C_CURV) * safe_norm(ub)))
                if var["fixed_radius"]:
                    floor = 0.0 * floor
                loss = loss + cfg["w_ent"] * (l_ent + floor)
                logs["ent"] = l_ent
                logs["floor"] = floor
            if var["use_con"]:
                da, db, du = pdist(ua, ua), pdist(ub, ub), pdist(u1, u1)
                off = 1.0 - jnp.eye(bsz)
                l_con = jnp.sum((jax.nn.relu(da - db) + jax.nn.relu(db - du)) * off) / jnp.sum(off)
                loss = loss + cfg["w_con"] * l_con
                logs["con"] = l_con
        return loss, logs

    return loss_fn, embed


# --------------------------------------------------------------------------- evaluation
def hungarian_v2(y_true, y_pred, old_mask):
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for p, t in zip(y_pred, y_true):
        w[p, t] += 1
    r, c = linear_sum_assignment(w.max() - w)
    mapping = dict(zip(c, r))  # true -> pred
    correct = np.array([mapping.get(t, -1) == p for p, t in zip(y_pred, y_true)])
    return correct.mean(), correct[old_mask].mean(), correct[~old_mask].mean()


def ss_kmeans(feat, lab_mask, y, n_clusters, known_classes, seed, iters=100):
    rng = np.random.default_rng(seed)
    lab_idx = np.where(lab_mask)[0]
    unl_idx = np.where(~lab_mask)[0]
    cls_to_k = {k: i for i, k in enumerate(known_classes)}
    lab_k = np.array([cls_to_k[t] for t in y[lab_idx]])
    cent = np.zeros((n_clusters, feat.shape[1]))
    for i in range(len(known_classes)):
        cent[i] = feat[lab_idx[lab_k == i]].mean(0)
    pool = feat[unl_idx]
    for j in range(len(known_classes), n_clusters):   # k-means++ on unlabeled
        d2 = ((pool[:, None, :] - cent[None, :j]) ** 2).sum(-1).min(1)
        cent[j] = pool[rng.choice(len(pool), p=d2 / d2.sum())]
    assign = np.zeros(len(feat), dtype=int)
    assign[lab_idx] = lab_k
    for _ in range(iters):
        d2 = ((feat[unl_idx][:, None, :] - cent[None]) ** 2).sum(-1)
        new = d2.argmin(1)
        changed = np.any(new != assign[unl_idx])
        assign[unl_idx] = new
        for k in range(n_clusters):
            m = assign == k
            if m.any():
                cent[k] = feat[m].mean(0)
        if not changed:
            break
    return assign


def evaluate(params, embed_fn, data, seed):
    X, Y, P, known, lab = data["X"], data["Y"], data["P"], data["known"], data["lab"]
    feats, zs = [], []
    bb = jax.jit(lambda p, x: backbone(p, x))
    emb = jax.jit(embed_fn)
    for i in range(0, len(X), 512):
        feats.append(np.asarray(bb(params, jnp.asarray(X[i:i + 512]))))
        zs.append(np.asarray(emb(params, jnp.asarray(X[i:i + 512]))))
    h = np.concatenate(feats)
    h = h / np.linalg.norm(h, axis=1, keepdims=True)
    known_classes = sorted(np.unique(Y[known]).tolist())
    assign = ss_kmeans(h, lab, Y, N_PARENT * N_CHILD, known_classes, seed)
    unl = ~lab
    allacc, oldacc, newacc = hungarian_v2(Y[unl], assign[unl], known[unl])
    # parent-level structure (unsupervised k-means with 8 clusters)
    km = KMeans(N_PARENT, n_init=10, random_state=seed).fit(h[unl])
    par_acc = hungarian_v2(P[unl], km.labels_, np.ones(unl.sum(), bool))[0]
    # novel -> known-sibling attribution through labeled prototypes
    protos = np.stack([h[lab & (Y == k)].mean(0) for k in known_classes])
    proto_parent = np.array([k // N_CHILD for k in known_classes])
    nov = unl & ~known
    nearest = ((h[nov][:, None, :] - protos[None]) ** 2).sum(-1).argmin(1)
    sib_acc = float(np.mean(proto_parent[nearest] == P[nov]))
    # validation accuracy on held-out KNOWN-class images (nearest labeled prototype)
    hv = np.asarray(bb(params, jnp.asarray(data["Xval"])))
    hv = hv / np.linalg.norm(hv, axis=1, keepdims=True)
    pred = np.array(known_classes)[((hv[:, None, :] - protos[None]) ** 2).sum(-1).argmin(1)]
    val_acc = float(np.mean(pred == data["Yval"]))
    # validation CLUSTERING accuracy on held-out known-class images (GCD model-selection protocol):
    # k-means with one cluster per known class, Hungarian-matched to the true labels; no novel labels used
    kmv = KMeans(len(known_classes), n_init=10, random_state=seed).fit(hv)
    val_cluster = float(hungarian_v2(data["Yval"], kmv.labels_, np.ones(len(hv), bool))[0])
    return dict(all=float(allacc), old=float(oldacc), new=float(newacc),
                parent_acc=float(par_acc), novel_to_parent=sib_acc, val_known=val_acc,
                val_cluster=val_cluster), np.concatenate(zs)


# --------------------------------------------------------------------------- training
DEFAULT_CFG = dict(epochs=40, batch=192, lr=2e-3, wd=1e-4,
                   f_a=1.5, f_b=3.0, tau_a=0.5, tau_b=0.3,
                   w_dir=0.5, w_ent=1.0, w_con=0.05, eps_floor=0.15, r_a=0.12, r_b=0.2,
                   w_fourex=0.1, w_naive=0.5)


def run(method, seed, cfg, out_dir, tag=""):
    data = make_dataset(seed)
    key = jax.random.PRNGKey(seed)
    params = init_params(key)
    n = len(data["X"])
    steps_per_epoch = n // cfg["batch"]
    total = cfg["epochs"] * steps_per_epoch
    sched = optax.cosine_decay_schedule(cfg["lr"], total)
    opt = optax.adamw(sched, weight_decay=cfg["wd"])
    state = opt.init(params)
    loss_fn, embed_fn = make_loss(method, cfg)

    @jax.jit
    def step(params, state, batch, alpha, key):
        (loss, logs), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch, alpha, key)
        updates, state = opt.update(grads, state, params)
        return optax.apply_updates(params, updates), state, loss, logs

    rng = np.random.default_rng(seed + 123)
    y_known = data["Y"].copy()
    t0 = time.time()
    it = 0
    history = []
    for ep in range(cfg["epochs"]):
        order = rng.permutation(n)
        for s in range(steps_per_epoch):
            idx = order[s * cfg["batch"]:(s + 1) * cfg["batch"]]
            xb = data["X"][idx]
            batch = dict(x1=jnp.asarray(augment(rng, xb)), x2=jnp.asarray(augment(rng, xb)),
                         y=jnp.asarray(np.where(data["lab"][idx], y_known[idx], -1)),
                         lab=jnp.asarray(data["lab"][idx]))
            alpha = jnp.asarray(min(1.0, it / max(total - 1, 1)), jnp.float32)
            key, sub = jax.random.split(key)
            params, state, loss, logs = step(params, state, batch, alpha, sub)
            it += 1
        history.append({k: float(v) for k, v in logs.items()} | {"loss": float(loss), "epoch": ep})
    train_time = time.time() - t0
    metrics, _ = evaluate(params, embed_fn, data, seed)
    # depth ordering diagnostics (hyperbolic methods)
    if method != "gcd":
        xs = jnp.asarray(data["X"][:1024])
        e = jax.jit(embed_fn)
        uu = e(params, xs)
        var = FREQCONE_VARIANTS.get(method)
        if var is not None and var["fixed_radius"]:
            fixed = jax.jit(lambda p, x, r: (lambda z: z / safe_norm(z, keepdims=True) * (r / jnp.sqrt(C_CURV)))(head(p, backbone(p, x))))
            ua, ub = fixed(params, lowpass(xs, cfg["f_a"]), cfg["r_a"]), fixed(params, lowpass(xs, cfg["f_b"]), cfg["r_b"])
        else:
            ua, ub = e(params, lowpass(xs, cfg["f_a"])), e(params, lowpass(xs, cfg["f_b"]))
        dep = lambda u: np.asarray(2 / np.sqrt(C_CURV) * np.arctanh(np.sqrt(C_CURV) * np.linalg.norm(np.asarray(u), axis=1)))
        da, db, du = dep(ua), dep(ub), dep(uu)
        ubn, uun = np.asarray(ub), np.asarray(uu)
        cosang = np.sum(ubn * uun, 1) / (np.linalg.norm(ubn, axis=1) * np.linalg.norm(uun, axis=1))
        metrics.update(in_cone_b_u=float(np.mean(np.asarray(cone_energy(ub, uu)) < 1e-6)),
                       angle_b_u_deg=float(np.degrees(np.arccos(np.clip(cosang, -1, 1))).mean()))
        metrics.update(depth_a=float(da.mean()), depth_b=float(db.mean()), depth_full=float(du.mean()),
                       frac_ordered=float(np.mean((da <= db) & (db <= du))))
    metrics.update(method=method + tag, seed=seed, train_time_s=round(train_time, 1), w_ent=cfg["w_ent"])
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{method}{tag}_seed{seed}.json"), "w") as f:
        json.dump(dict(metrics=metrics, cfg=cfg, history=history[-3:]), f, indent=1)
    print(json.dumps(metrics))
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["gcd", "hypcd", "fourex_partition", "naive_lowpass",
                             "freqcone_v0", "freqcone_v1", "freqcone", "freqcone_no_con", "freqcone_no_ent",
                             "freqcone_v3"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--w_ent", type=float, default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = dict(DEFAULT_CFG)
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.w_ent is not None:
        cfg["w_ent"] = args.w_ent
    run(args.method, args.seed, cfg, args.out, args.tag)
