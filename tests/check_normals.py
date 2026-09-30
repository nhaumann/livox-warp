"""Voxel-summary normals against exact per-point PCA on a real recording.

The pipeline estimates normals from a per-voxel summary of the cloud (Pipeline.neighborhoods) instead of
a PCA over every point's neighbours. On any recording (LIVOX_RECORDING, or the first *.lvxr under
$LIVOX_DATA_DIR/recordings) the median angle between the two estimates must stay under 10 degrees; skips
when there is no recording. Only the first few seconds are used: both estimators see the same cloud, so a
moving scanner (whose sensor-frame points smear) is as good a test as a still one.
"""

import sys

import common
import numpy as np
import warp as wp
from livox_warp import gpu
from livox_warp.sources import clean_returns

RADIUS = 0.10
SECONDS = 10.0
MAX_MEDIAN_DEG = 10.0


@wp.kernel
def exact(grid: wp.uint64, xyz: wp.array(dtype=wp.vec3), radius: float, out: wp.array(dtype=wp.vec3)):
    """The smallest eigenvector of the covariance of every neighbour within `radius`."""
    i = wp.tid()
    p = xyz[i]
    q = wp.hash_grid_query(grid, p, radius)
    j = int(0)
    c = int(0)
    s1 = wp.vec3()
    s2 = wp.mat33()
    while wp.hash_grid_query_next(q, j):
        d = xyz[j] - p
        if wp.dot(d, d) <= radius * radius:
            c += 1
            s1 += d
            s2 += wp.outer(d, d)
    mu = s1 / float(c)
    Q, ev = wp.eig3(s2 / float(c) - wp.outer(mu, mu))
    k = int(0)
    if ev[1] < ev[k]:
        k = 1
    if ev[2] < ev[k]:
        k = 2
    out[i] = wp.vec3(Q[0, k], Q[1, k], Q[2, k])


def main():
    rec = common.any_recording()
    if rec is None:
        common.skip("no recording", "LIVOX_RECORDING")
    dev = common.device()
    xyz, attr, t = common.load_recording(rec)
    keep = clean_returns(xyz, attr) & (t - t.min() < SECONDS)
    pipe = gpu.Pipeline(ring_capacity=1 << 21, map_capacity=1 << 20, device=dev)
    P = np.ascontiguousarray(xyz[keep][: pipe.work_cap])
    n = len(P)

    pts = wp.array(P, dtype=wp.vec3, device=dev)
    grid = wp.HashGrid(128, 128, 128, device=dev)
    grid.build(pts, RADIUS)
    ex = wp.zeros(n, dtype=wp.vec3, device=dev)
    wp.launch(exact, dim=n, inputs=[grid.id, pts, RADIUS, ex], device=dev)

    pipe.load_points(P)
    pipe.neighborhoods(RADIUS, True, (0, 0, 0))
    a, b = ex.numpy(), pipe.nrm.numpy()[:n]
    norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    cos = np.abs((a * b).sum(1)) / np.maximum(norms, 1e-12)
    ang = np.degrees(np.arccos(np.clip(cos, 0, 1)))
    med, p90 = float(np.median(ang)), float(np.percentile(ang, 90))
    print(f"{rec.name}: {n:,} points, voxel vs exact normal angle median {med:.1f} deg, p90 {p90:.1f} deg")
    ok = med < MAX_MEDIAN_DEG
    print("OK" if ok else f"FAIL: median normal angle {med:.1f} deg, limit {MAX_MEDIAN_DEG:.0f} deg")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
