"""Calibrate the Mid-40 against a prior scan: range bias and scale, noise, and angular distortion.

usage: python tools/calib_prior_map.py [recording.lvxr ...] [--map prior.npz]

Each recording of the scanner standing still inside the mapped area (default: LIVOX_STATIC_RECORDINGS, see
tests/README.md) is localised in the scan (then refined with all its points), and every return that lands
on an unchanged surface (within 10 cm of the scan, on a planar scan point) is compared with that surface:
its signed distance along the scan's normal (positive = in front of the surface, toward the sensor) and,
divided by the cosine of the incidence angle, its range error along the ray (positive = measured long).
Reported by range, incidence, reflectivity, return (first / second) and position in the 38 deg field of
view, with a robust fit range_error = offset + scale * range. A terrestrial scanner's own error (~2-3 mm)
is small next to the Mid-40's (~2 cm), so what is left is the Mid-40.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import common  # noqa: E402
from livox_warp import localize, prior_map  # noqa: E402
from livox_warp.sources import clean_returns  # noqa: E402


def quiet(*_):
    """A log sink for the localizer."""


def robust(v):
    """median and 1.4826 * MAD (a sigma that ignores outliers)."""
    if len(v) == 0:
        return float("nan"), float("nan")
    m = float(np.median(v))
    return m, 1.4826 * float(np.median(np.abs(v - m)))


def table(title, key, edges, fmt, d_mm, e_mm, min_n=500):
    print(f"  {title}")
    print(f"    {'bin':>14s} {'points':>9s}  {'normal bias':>11s} {'sigma':>7s}  "
          f"{'range bias':>10s} {'sigma':>7s}  (mm)")
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (key >= lo) & (key < hi)
        if s.sum() < min_n:
            continue
        bn, sn = robust(d_mm[s])
        br, sr = robust(e_mm[s])
        print(f"    {fmt(lo, hi):>14s} {s.sum():9,d}  {bn:11.1f} {sn:7.1f}  {br:10.1f} {sr:7.1f}")


def residuals(path, L, M, N, planar):
    """One recording's returns against the scan: a dict of per-return arrays, or None if it could not be
    localised uniquely."""
    t0 = time.perf_counter()
    x, a, t = common.load_recording(path)
    clean = clean_returns(x, a, min_range=0.5)
    x, a, t = x[clean], a[clean], t[clean]
    first = t - t.min() < 1.5
    T, info = L.search(x[first], log=quiet)
    if not info["unique"]:
        print(f"{path.name}: localisation not unique, skipped")
        return None
    T, fit, _ = L.refine(x, T, schedule=((0.10, 8), (0.05, 10), (0.03, 10)))
    c = T[:3, 3]
    near = np.linalg.norm(M - c, axis=1) < 40.0
    tree = cKDTree(M[near])
    Mn, Nn, Pn = M[near], N[near], planar[near]
    w = x.astype(np.float64) @ T[:3, :3].T + c
    dist, j = tree.query(w, distance_upper_bound=0.10)
    ok = np.isfinite(dist)
    j = np.where(ok, j, 0)
    ok &= Pn[j] > 0.6
    q, n = Mn[j], Nn[j].copy()
    flip = np.einsum("ij,ij->i", n, c - q) < 0  # orient the normal toward the sensor
    n[flip] *= -1
    ray = w - c
    rng = np.linalg.norm(ray, axis=1)
    u = ray / rng[:, None]
    cosi = -np.einsum("ij,ij->i", u, n)  # cos of the incidence angle
    d = np.einsum("ij,ij->i", w - q, n)  # signed: + in front of the surface
    ok &= cosi > 0.2
    e = -d / np.maximum(cosi, 0.2)  # along the ray: + measured long
    off_axis = np.degrees(np.arccos(np.clip(x[:, 0] / np.linalg.norm(x, axis=1), -1, 1)))
    refl = (a & 0xFF).astype(np.int32)
    ret = ((a >> 16) & 0xFF).astype(np.int32)
    print(f"{path.name}: {len(x):,} returns, {ok.mean():.0%} on unchanged planar surfaces; "
          f"fit {fit[0]:.2f} within 3 cm after refinement ({time.perf_counter() - t0:.0f} s)")
    return dict(rng=rng[ok], cosi=cosi[ok], d=d[ok] * 1000, e=e[ok] * 1000, off=off_axis[ok],
                refl=refl[ok], ret=ret[ok])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("recordings", nargs="*", type=Path,
                    help="recordings of the scanner standing still (default: LIVOX_STATIC_RECORDINGS)")
    ap.add_argument("--map", type=Path, default=common.prior_map(), help="the prior map (default: LIVOX_PRIOR_MAP)")
    args = ap.parse_args()
    recs = args.recordings or common.static_recordings()
    if args.map is None or not recs:
        common.skip("needs the prior map and recordings of the scanner standing still inside it",
                    "LIVOX_PRIOR_MAP", "LIVOX_STATIC_RECORDINGS")
    dev = common.device()
    pm = prior_map.load(str(args.map))
    L = localize.Localizer(pm["xyz"], pm["normal"], pm.get("planarity"), device=dev)
    M = pm["xyz"].astype(np.float64)
    N = pm["normal"].astype(np.float64)
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
    planar = pm["planarity"].astype(np.float32)
    rows = []
    for path in recs:
        if not path.exists():
            print(f"{path.name}: not found, skipped")
            continue
        r = residuals(path, L, M, N, planar)
        if r is not None:
            rows.append(r)
    if not rows:
        sys.exit("nothing to calibrate against")
    R = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    d, e = R["d"], R["e"]
    print(f"\nall: {len(d):,} returns; normal bias {np.median(d):.1f} mm, sigma {robust(d)[1]:.1f} mm; "
          f"range bias {np.median(e):.1f} mm, sigma {robust(e)[1]:.1f} mm")
    good = R["cosi"] > 0.7  # near head-on: the range error dominates the normal residual
    A = np.stack([np.ones(good.sum()), R["rng"][good]], 1)
    wgt = np.ones(good.sum())
    coef = np.zeros(2)
    for _ in range(10):  # iteratively reweighted least squares (Huber, 20 mm)
        coef = np.linalg.lstsq(A * wgt[:, None], e[good] * wgt, rcond=None)[0]
        r = e[good] - A @ coef
        wgt = np.sqrt(np.minimum(1.0, 20.0 / np.maximum(np.abs(r), 1e-6)))
    print(f"range error ~ {coef[0]:.1f} mm + {coef[1]:.2f} mm/m * range (incidence < 45 deg, {good.sum():,} returns)\n")
    table("by range", R["rng"], [0.5, 1, 2, 3, 4, 6, 8, 12, 20], lambda lo, hi: f"{lo:g}-{hi:g} m", d, e)
    inc = np.degrees(np.arccos(np.clip(R["cosi"], -1, 1)))
    table("by incidence angle", inc, [0, 15, 30, 45, 60, 75, 80], lambda lo, hi: f"{lo:g}-{hi:g} deg", d, e)
    table("by reflectivity", R["refl"], [0, 5, 10, 20, 40, 80, 150, 200, 256],
          lambda lo, hi: f"{lo}-{hi - 1}", d, e)
    table("by return", R["ret"], [0, 1, 2, 3], lambda lo, hi: ["first", "second", "third"][lo], d, e)
    table("by angle off the optical axis (the rosette's radius)", R["off"], [0, 4, 8, 12, 16, 19.5],
          lambda lo, hi: f"{lo:g}-{hi:g} deg", d, e)
    print("\n  (sigma along the normal is what registration sees; the odometry's noise model is Odometry.noise)")


if __name__ == "__main__":
    main()
