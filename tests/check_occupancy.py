"""The differentiable occupancy field (occupancy.py).

1. The tape's gradient of a few rays' negative log-likelihood matches central differences on the cells it depends
   on (few rays: float32 sums of many would drown the differences).
2. Dual returns: firings that return at 3 m and again at 5 m make the 3 m layer stop half the beam (its window's
   stopping probability settles at 1/2, the likelihood's optimum for one echo stopping and one passing); firings
   that return at 3 m only make it stop all of it. The free space in front stays free.
3. Changes on the simulated office: the scanner stations are estimated from the scan's point density (within
   15 cm of where the simulated scanner stood); trained on the walk's rays (ground-truth poses) from the scan's
   occupancy, the field marks the added box ADDED and the removed one REMOVED, few cells elsewhere, and the
   surfaces no station could see UNSCANNED.
"""

import math

import common
import numpy as np
from livox_warp import dmath, localize, occupancy, prior_map


def cone(n, half_deg, rng):
    """n unit directions within half_deg of +x."""
    a = np.radians(half_deg) * np.sqrt(rng.random(n))
    b = 2 * math.pi * rng.random(n)
    d = np.stack([np.cos(a), np.sin(a) * np.cos(b), np.sin(a) * np.sin(b)], 1)
    return d.astype(np.float32)


def gradient_check(dev):
    rng = np.random.default_rng(0)
    f = occupancy.OccupancyField((-0.5, -1.0, -1.0), (6.0, 1.0, 1.0), 0.05, device=dev)
    f.warm_start(None)
    theta = f.theta.numpy()
    theta[:] = rng.normal(-3.0, 2.0, f.n)  # a field with something everywhere: every sample matters
    f.theta.assign(theta)
    n = 6
    d = cone(n, 5.0, rng)
    r = np.array([3.0, 3.5, 4.0, 4.5, -1.0, 5.0], np.float32)  # one firing that returned nothing
    f.set_rays(np.zeros((n, 3), np.float32), d, r)
    ids = np.arange(n)
    loss0, grad = f.loss_and_grad(ids, s_free=24, s_hit=8, seed=5)
    worst = 0.0
    for c in np.argsort(-np.abs(grad))[:8]:
        vals = []
        for sgn in (1, -1):
            th = theta.copy()
            th[c] += sgn * 0.01
            f.theta.assign(th)
            vals.append(f.loss_and_grad(ids, s_free=24, s_hit=8, seed=5)[0])
        fd = (vals[0] - vals[1]) / 0.02
        rel = abs(fd - grad[c]) / max(abs(fd), abs(grad[c]), 1e-9)
        worst = max(worst, rel)
        assert rel < 0.03, f"cell {c}: tape {grad[c]:.5g}, finite differences {fd:.5g}"
    f.theta.assign(theta)
    print(f"gradients: tape vs central differences on the 8 cells that matter most: worst {worst:.2%} apart")


def window_stop(f, d, r, half=0.06, step=0.005):
    """Probability that a ray along d stops within r +- half, from the field's density."""
    s = np.arange(r - half, r + half, step) + step / 2
    occ = np.clip(f.occupancy_at(np.outer(s, d)), 0.0, 1.0 - 1e-6)
    tau = np.sum(-np.log1p(-occ) / f.cell * step)  # per-cell stopping probability -> density -> optical depth
    return 1.0 - math.exp(-tau)


def dual_returns(dev):
    rng = np.random.default_rng(1)
    n = 40_000
    d = cone(n, 4.0, rng)
    out = {}
    for case, ranges in (("single", (3.0,)), ("dual", (3.0, 5.0))):
        f = occupancy.OccupancyField((-0.5, -0.6, -0.6), (6.0, 0.6, 0.6), 0.05, device=dev)
        f.warm_start(None)
        o = np.zeros((n * len(ranges), 3), np.float32)
        dd = np.concatenate([d] * len(ranges))
        rr = np.concatenate([np.full(n, x, np.float32) for x in ranges])
        f.set_rays(o, dd, rr)
        f.train(epochs=12, batch=1 << 15, lr=(0.3, 0.02), prior_weight=1e-4)
        axis = np.array([1.0, 0.0, 0.0])
        out[case] = (window_stop(f, axis, 3.0), window_stop(f, axis, 5.0), window_stop(f, axis, 1.5, half=0.5))
    s3, _, s_free = out["single"]
    d3, d5, d_free = out["dual"]
    print(f"dual returns: the 3 m layer stops {s3:.2f} of the beam when nothing comes back from behind it, "
          f"{d3:.2f} when a second echo comes from 5 m (which stops {d5:.2f}); the metre in front stops "
          f"{max(s_free, d_free):.3f}")
    assert s3 > 0.95, s3
    assert 0.35 < d3 < 0.65, d3
    assert d5 > 0.9, d5
    assert max(s_free, d_free) < 0.05


def changes(dev):
    pm = common.sim_prior_map(dev)
    stations = prior_map.estimate_stations(pm)
    truth = np.array(common.TLS_STATIONS)
    off = np.array([np.min(np.linalg.norm(stations - s, axis=1)) for s in truth])
    print(f"stations: {len(stations)} estimated from the scan's point density, the true ones {off.max() * 100:.0f} cm "
          f"off at most")
    assert len(stations) == len(truth) and off.max() < 0.15, (stations, off)
    loc = localize.Localizer(pm["xyz"], pm["normal"], pm["planarity"], device=dev)
    sim, x, a, t = common.sim_walk(dev, 20.0)
    times = np.arange(0.0, 20.05, 0.01)
    T = dmath.interpolate_poses(times, np.array([sim.pose_at(float(s)) for s in times]), t)
    r = np.linalg.norm(x, axis=1)
    o = T[:, :3, 3].astype(np.float32)
    d = np.einsum("nij,nj->ni", T[:, :3, :3], x / r[:, None]).astype(np.float32)
    f = occupancy.OccupancyField.around(o, d, r, 0.05, device=dev)
    f.warm_start(loc.grid, stations)
    f.set_rays(o, d, r.astype(np.float32))
    f.train(epochs=6, log=print)
    counts = f.classify()
    print(f"changes: {counts}")
    lane = lambda p: (p[:, 0] > 11.3) & (p[:, 0] < 12.7)  # noqa: E731 - where the person walks
    added = f.centres(occupancy.ADDED)
    removed = f.centres(occupancy.REMOVED)
    added, removed = added[~lane(added)], removed[~lane(removed)]
    on_a = common.box_distance(added, common.ADDED_BOX) < 0.1
    on_r = common.box_distance(removed, common.REMOVED_BOX) < 0.1
    print(f"         {on_a.sum()} added cells on the added box, {(~on_a).sum()} elsewhere; {on_r.sum()} removed cells "
          f"on the removed box, {(~on_r).sum()} elsewhere (the person's lane left out)")
    assert on_a.sum() >= 100 and on_a.mean() >= 0.3, (on_a.sum(), on_a.mean())
    assert on_r.sum() >= 100 and on_r.mean() >= 0.3, (on_r.sum(), on_r.mean())


def main():
    dev = common.device()
    gradient_check(dev)
    dual_returns(dev)
    changes(dev)


if __name__ == "__main__":
    main()
