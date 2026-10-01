"""Differentiable building blocks shared by the solvers (walkfit, occupancy).

Warp differentiates kernels with wp.Tape: launches recorded inside `with tape:` get generated adjoint kernels,
and tape.backward(loss=...) fills the .grad of every array created with requires_grad=True. Two rules keep
those gradients exact here:
  - loops that accumulate a differentiable value have a compile-time trip count (unrolled), never a runtime one;
    where a sum runs over a runtime count, each term gets its own thread and an atomic add instead;
  - constants the loss depends on (associations, base poses) are written by kernels outside the tape.

Also here: the uniform cubic B-spline basis, SO(3) exponential, Geman-McClure and softplus, an Adam step kernel
(per-element masks, so a sliding window can freeze part of a trajectory), and host-side pose interpolation.
"""

from __future__ import annotations

import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation, Slerp


@wp.func
def bspline4(u: float) -> wp.vec4:
    """Uniform cubic B-spline weights of control points k-1, k, k+1, k+2 at u in [0, 1) of segment k."""
    u2 = u * u
    u3 = u2 * u
    return wp.vec4((1.0 - 3.0 * u + 3.0 * u2 - u3) / 6.0,
                   (4.0 - 6.0 * u2 + 3.0 * u3) / 6.0,
                   (1.0 + 3.0 * u + 3.0 * u2 - 3.0 * u3) / 6.0,
                   u3 / 6.0)


@wp.func
def so3_exp(phi: wp.vec3) -> wp.mat33:
    """Rotation matrix of a rotation vector (Rodrigues), with a series near zero so the gradient stays finite."""
    th2 = wp.dot(phi, phi)
    K = wp.skew(phi)
    eye = wp.identity(n=3, dtype=wp.float32)
    if th2 < 1.0e-8:
        return eye + K + 0.5 * (K * K)
    th = wp.sqrt(th2)
    return eye + (wp.sin(th) / th) * K + ((1.0 - wp.cos(th)) / th2) * (K * K)


@wp.func
def geman_mcclure(r: float, sigma: float) -> float:
    """sigma^2 r^2 / (r^2 + sigma^2): r^2 near zero, bounded at sigma^2 for outliers."""
    r2 = r * r
    s2 = sigma * sigma
    return s2 * r2 / (r2 + s2)


@wp.func
def softplus(x: float) -> float:
    """log(1 + e^x), computed without overflow."""
    return wp.max(x, 0.0) + wp.log(1.0 + wp.exp(-wp.abs(x)))


@wp.func
def quat_interp(q0: wp.quat, q1: wp.quat, u: float) -> wp.quat:
    """Slerp along the shorter arc (no gradient needed: used for constant base poses only)."""
    a = wp.vec4(q0[0], q0[1], q0[2], q0[3])
    b = wp.vec4(q1[0], q1[1], q1[2], q1[3])
    d = wp.dot(a, b)
    if d < 0.0:
        b = -b
        d = -d
    out = wp.vec4()
    if d > 0.9995:
        out = wp.normalize(a + (b - a) * u)
    else:
        th = wp.acos(d)
        s = wp.sin(th)
        out = (wp.sin((1.0 - u) * th) / s) * a + (wp.sin(u * th) / s) * b
    return wp.quat(out[0], out[1], out[2], out[3])


@wp.kernel
def k_adam_vec3(
    p: wp.array(dtype=wp.vec3),
    g: wp.array(dtype=wp.vec3),
    m: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    mask: wp.array(dtype=float),
    lr: float,
    b1: float,
    b2: float,
    c1: float,
    c2: float,
    eps: float,
):
    """One Adam step on vec3 parameters; mask[i] = 0 freezes element i (its moments are left alone)."""
    i = wp.tid()
    if mask[i] == 0.0:
        return
    gi = g[i]
    mi = b1 * m[i] + (1.0 - b1) * gi
    vi = b2 * v[i] + (1.0 - b2) * wp.cw_mul(gi, gi)
    m[i] = mi
    v[i] = vi
    step = wp.vec3(mi[0] / (wp.sqrt(vi[0] / c2) + eps),
                   mi[1] / (wp.sqrt(vi[1] / c2) + eps),
                   mi[2] / (wp.sqrt(vi[2] / c2) + eps))
    p[i] = p[i] - (mask[i] * lr / c1) * step


@wp.kernel
def k_adam_f32(
    p: wp.array(dtype=float),
    g: wp.array(dtype=float),
    m: wp.array(dtype=float),
    v: wp.array(dtype=float),
    mask: wp.array(dtype=float),
    lr: float,
    b1: float,
    b2: float,
    c1: float,
    c2: float,
    eps: float,
):
    """One Adam step on scalar parameters; mask[i] = 0 freezes element i."""
    i = wp.tid()
    if mask[i] == 0.0:
        return
    gi = g[i]
    mi = b1 * m[i] + (1.0 - b1) * gi
    vi = b2 * v[i] + (1.0 - b2) * gi * gi
    m[i] = mi
    v[i] = vi
    p[i] = p[i] - mask[i] * lr * (mi / c1) / (wp.sqrt(vi / c2) + eps)


class Adam:
    """Adam over one array of float or vec3 parameters (the array must have requires_grad=True).

    step(lr, mask) applies one update from p.grad, mask a float array (0 freezes an element; for scalars it
    defaults to all ones); reset() forgets the moments (a new stage of the objective)."""

    def __init__(self, p: wp.array, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-12):
        self.p = p
        self.b1, self.b2, self.eps = b1, b2, eps
        self.m = wp.zeros_like(p, requires_grad=False)
        self.v = wp.zeros_like(p, requires_grad=False)
        self.t = 0
        self.vec3 = p.dtype == wp.vec3
        self.ones = None if self.vec3 else wp.ones(p.shape[0], dtype=float, device=p.device)

    def reset(self):
        self.m.zero_()
        self.v.zero_()
        self.t = 0

    def step(self, lr: float, mask: wp.array | None = None):
        self.t += 1
        c1 = 1.0 - self.b1**self.t
        c2 = 1.0 - self.b2**self.t
        d = self.p.device
        if self.vec3:
            if mask is None:
                raise ValueError("vec3 parameters take a mask (ones to update everything)")
            wp.launch(k_adam_vec3, dim=self.p.shape[0], device=d,
                      inputs=[self.p, self.p.grad, self.m, self.v, mask, lr, self.b1, self.b2, c1, c2, self.eps])
        else:
            wp.launch(k_adam_f32, dim=self.p.shape[0], device=d,
                      inputs=[self.p, self.p.grad, self.m, self.v, self.ones if mask is None else mask, lr, self.b1,
                              self.b2, c1, c2, self.eps])


def cosine_lr(lr0: float, lr1: float, i: int, n: int) -> float:
    """Learning rate at step i of n, from lr0 down to lr1 along half a cosine."""
    return lr1 + 0.5 * (lr0 - lr1) * (1.0 + np.cos(np.pi * min(i, n) / max(n, 1)))


# ---- host-side poses ----------------------------------------------------------------------------


def interpolate_poses(times: np.ndarray, poses: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Poses (4x4) at times t by slerp / linear interpolation of the keyed poses (clamped at the ends)."""
    times = np.asarray(times, dtype=np.float64)
    t = np.clip(np.asarray(t, dtype=np.float64), times[0], times[-1])
    out = np.tile(np.eye(4), (len(t), 1, 1))
    if len(times) == 1:
        out[:] = poses[0]
        return out
    out[:, :3, :3] = Slerp(times, Rotation.from_matrix(poses[:, :3, :3]))(t).as_matrix()
    for a in range(3):
        out[:, a, 3] = np.interp(t, times, poses[:, a, 3])
    return out


def rotvec_to_matrix(phi: np.ndarray) -> np.ndarray:
    return Rotation.from_rotvec(np.asarray(phi, dtype=np.float64)).as_matrix()
