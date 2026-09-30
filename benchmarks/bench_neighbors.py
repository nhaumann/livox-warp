"""Neighbour-pass cost on a dense accumulated cloud: exact per-point counting vs the voxel summary.

Takes any recording (LIVOX_RECORDING, or the first *.lvxr under $LIVOX_DATA_DIR/recordings), stacks
jittered copies of its points to --points (default 4.4 M: what a few seconds of returns accumulate on one
room) and times the plain per-point neighbour count against Pipeline.neighborhoods, which summarises the
cloud per voxel first. Not a test: it prints the timings, how well the weighted count agrees with the
exact one, and how flat the normals on the floor come out.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import common  # noqa: E402
from livox_warp import gpu  # noqa: E402


@wp.kernel
def k_count_all(grid: wp.uint64, xyz: wp.array(dtype=wp.vec3), radius: float, out: wp.array(dtype=wp.int32)):
    """The plain pass: every point visits every neighbour."""
    i = wp.tid()
    p = xyz[i]
    q = wp.hash_grid_query(grid, p, radius)
    j = int(0)
    c = int(0)
    while wp.hash_grid_query_next(q, j):
        if wp.length(xyz[j] - p) <= radius:
            c += 1
    out[i] = c


def timed(label, fn, reps=3):
    fn()
    wp.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    wp.synchronize()
    ms = (time.perf_counter() - t0) / reps * 1e3
    print(f"{label:<40} {ms:9.1f} ms")
    return ms


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--points", type=int, default=4_400_000, help="size of the stacked cloud (default 4.4 M)")
    ap.add_argument("--radius", type=float, default=0.10, help="neighbourhood radius in metres (default 0.1)")
    args = ap.parse_args()
    rec = common.any_recording()
    if rec is None:
        common.skip("no recording", "LIVOX_RECORDING")
    dev = common.device()
    base = common.load_recording(rec)[0]
    rng = np.random.default_rng(0)
    reps = int(np.ceil(args.points / len(base)))
    dense = np.concatenate([base + rng.normal(0, 0.002, base.shape).astype(np.float32) for _ in range(reps)])
    dense = np.ascontiguousarray(dense[: args.points])
    n, radius = len(dense), args.radius
    print(f"{rec.name}: {n:,} points stacked from {len(base):,} recorded")

    pts = wp.array(dense, dtype=wp.vec3, device=dev)
    grid = wp.HashGrid(128, 128, 128, device=dev)
    cnt = wp.zeros(n, dtype=wp.int32, device=dev)
    grid.build(pts, radius)

    def exact():
        wp.launch(k_count_all, dim=n, inputs=[grid.id, pts, radius, cnt], device=dev)

    timed("exact (per-point): every neighbour", exact, reps=1)
    raw_cnt = cnt.numpy().copy()
    print(f"  median neighbours within {radius} m: {int(np.median(raw_cnt)):,}")

    # the pipeline path: voxel summary, weighted neighbours, gathered back to the points
    pipe = gpu.Pipeline(ring_capacity=1 << 23, map_capacity=1 << 20, device=dev)
    pipe.load_points(dense)
    timed("voxel summary: count only", lambda: pipe.neighborhoods(radius, False, (0, 0, 0)))
    print(f"  {pipe.nb_voxels:,} voxels")
    new_cnt = pipe.nbr.numpy()[:n]
    rel = np.abs(new_cnt - raw_cnt) / np.maximum(raw_cnt, 1)
    print(f"  weighted count vs exact: median rel. error {np.median(rel):.1%}")
    timed("voxel summary + normals", lambda: pipe.neighborhoods(radius, True, (0, 0, 0)))
    floor = dense[:, 2] < pipe.floor_z() + 0.1  # a low percentile of z: where the floor is
    nz = np.abs(pipe.nrm.numpy()[:n][floor, 2])
    print(f"  floor |nz| mean {nz.mean():.3f} over {floor.sum():,} floor points (1.0 = flat)")


if __name__ == "__main__":
    main()
