"""Check that the PyTorch FreqCone loss (the repository's single copy, freqcone/freqcone_torch.py) equals the JAX
loss used in the toy experiment.

Part 1 is the original check (v2, the configuration of the mechanism study; L_con happens to be 0 there).
Part 2 adds cases the original did not exercise: shallow and mid-depth points (L_con > 0), v3, and the
gradients of v2 and v3 in float64 (where the stop-gradient placement matters).
Run from anywhere: python mechanism_study/parity_check.py (needs jax, optax, torch).
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from freqcone import freqcone_torch as ft  # noqa: E402
import toy_experiment as te  # noqa: E402

cfg = te.DEFAULT_CFG
c = te.C_CURV


def jax_total(z, detach_child_norm=True):
    """Identical formulas to toy_experiment.make_loss for "freqcone" (v2) / "freqcone_v3"."""
    u1, u2 = te.to_ball(z[0]), te.to_ball(z[1])
    ua = z[2] / te.safe_norm(z[2], keepdims=True) * (cfg["r_a"] / jnp.sqrt(c))
    ub = z[3] / te.safe_norm(z[3], keepdims=True) * (cfg["r_b"] / jnp.sqrt(c))
    target = jax.lax.stop_gradient(u2)
    l_dir = te.cross_view_nce(te.cosine_matrix(ua, target), cfg["tau_a"]) + \
        te.cross_view_nce(te.cosine_matrix(ub, target), cfg["tau_b"])
    if detach_child_norm:
        n1 = te.safe_norm(u1, keepdims=True)
        child = jax.lax.stop_gradient(n1) * u1 / n1
    else:
        child = u1
    l_ent = jnp.mean(te.cone_energy(ua, ub) + te.cone_energy(ub, child))
    da, db, du = te.pdist(ua, ua), te.pdist(ub, ub), te.pdist(u1, u1)
    off = 1.0 - jnp.eye(z.shape[1])
    l_con = jnp.sum((jax.nn.relu(da - db) + jax.nn.relu(db - du)) * off) / jnp.sum(off)
    return cfg["w_dir"] * l_dir + cfg["w_ent"] * l_ent + cfg["w_con"] * l_con, (l_dir, l_ent, l_con)


def torch_loss(detach_child_norm=True):
    return ft.FreqConeLoss(c=c, K=te.K_CONE, r_a=cfg["r_a"], r_b=cfg["r_b"], tau_a=cfg["tau_a"], tau_b=cfg["tau_b"],
                           w_dir=cfg["w_dir"], w_ent=cfg["w_ent"], w_con=cfg["w_con"], detach_child_norm=detach_child_norm)


# ----------------------------------------------------------------------------- part 1 (original check)
rng = np.random.default_rng(0)
B, n = 48, 32
z = rng.normal(size=(4, B, n)).astype(np.float32) * 2.0
jt, (l_dir, l_ent, l_con) = jax_total(jnp.asarray(z))
t = [torch.tensor(z[i]) for i in range(4)]
torch_total, logs = torch_loss()(ft.to_ball(t[0], c, te.CLIP_R), ft.to_ball(t[1], c, te.CLIP_R), t[2], t[3])
print("JAX  :", float(jt), " dir", float(l_dir), " ent", float(l_ent), " con", float(l_con))
print("Torch:", float(torch_total), " dir", float(logs["fc_dir"]), " ent", float(logs["fc_ent"]), " con", float(logs["fc_con"]))
print("abs diff:", abs(float(jt) - float(torch_total)))
assert abs(float(jt) - float(torch_total)) < 1e-3

# ----------------------------------------------------------------------------- part 2 (added cases)
worst = 0.0
for scale in (0.05, 0.2, 2.0):                 # shallow (L_con > 0), mid, clipped
    for v3 in (False, True):
        zz = np.random.default_rng(1).normal(size=(4, 48, 32)).astype(np.float32) * scale
        jv, (_, _, jc) = jax_total(jnp.asarray(zz), detach_child_norm=not v3)
        tt = [torch.tensor(zz[i]) for i in range(4)]
        tv, tl = torch_loss(not v3)(ft.to_ball(tt[0], c, te.CLIP_R), ft.to_ball(tt[1], c, te.CLIP_R), tt[2], tt[3])
        worst = max(worst, abs(float(jv) - float(tv)))
        print(f"scale {scale:<4} {'v3' if v3 else 'v2'}: |loss diff| {abs(float(jv) - float(tv)):.2e}  "
              f"L_con {float(jc):.4f} / {float(tl['fc_con']):.4f}")
assert worst < 1e-3

jax.config.update("jax_enable_x64", True)
torch.set_default_dtype(torch.float64)
worst_grad = 0.0
for scale in (0.2, 2.0):
    for v3 in (False, True):
        zz = np.random.default_rng(2).normal(size=(4, 32, 16)) * scale
        gj = np.asarray(jax.grad(lambda x: jax_total(x, not v3)[0])(jnp.asarray(zz)))
        zt = torch.tensor(zz, requires_grad=True)
        tv, _ = torch_loss(not v3)(ft.to_ball(zt[0], c, te.CLIP_R), ft.to_ball(zt[1], c, te.CLIP_R), zt[2], zt[3])
        tv.backward()
        rel = float(np.abs(gj - zt.grad.numpy()).max() / (np.abs(gj).max() + 1e-30))
        worst_grad = max(worst_grad, rel)
        print(f"scale {scale:<4} {'v3' if v3 else 'v2'}: max relative gradient diff {rel:.2e}")
assert worst_grad < 1e-6
print(f"parity OK: worst |loss diff| {worst:.2e} (float32), worst relative gradient diff {worst_grad:.2e} (float64)")
