"""Headless check of the Warp pipeline: the simulated source through live and map modes, denoise and
normals, a pose-table entry moving the live window, and, when a recording is present, a real replay."""

import sys
import time

import common
import numpy as np
import warp as wp
from livox_warp import gpu
from livox_warp.sources import SIM_FLOOR_Z, ReplaySource, SimSource


def simulated(pipe, dev, out_pos, out_col, out_nrm):
    """Live and map modes on the simulated room, then a pose-table entry that moves the live window."""

    def live(now):
        return common.view(now, pipe, persist=0.5, voxel=0.03)

    sim = SimSource(device=dev)
    now = 0.0
    for _ in range(30):
        time.sleep(1 / 60)
        b = sim.poll()
        if b:
            xyz, attr, t, n = b
            now = sim.t
            pipe.ingest(xyz, attr, t, n, live(now), map_on=True, voxel=0.03)
    cnt = pipe.build(live(now), map_on=False, min_count=1, neighbors="normals", radius=0.15, sensor=(0, 0, 0))
    pipe.shade(common.shade(now, gpu.MODE_NORMAL, denoise=1, min_nbrs=4, has_normals=1), out_pos, out_col, out_nrm)
    wp.synchronize()
    alpha = (out_col.numpy()[:cnt] >> 24) & 0xFF
    print(f"sim live : {sim.emitted} emitted, {cnt} visible in 0.5 s window, denoise hid {np.mean(alpha == 0):.1%}")
    print("           ms", {k: round(v, 2) for k, v in pipe.ms.items()})
    nz = np.abs(pipe.nrm.numpy()[:cnt])
    floor = pipe.c_xyz.numpy()[:cnt][:, 2] < SIM_FLOOR_Z + 0.1
    print(f"           floor points {floor.sum()}, mean |nz| on floor {nz[floor, 2].mean():.3f} (expect ~1)")

    cnt = pipe.build(common.view(now, pipe, persist=0.0, voxel=0.03), map_on=True, min_count=1, neighbors="",
                     radius=0.1, sensor=(0, 0, 0))
    print(f"sim map  : {cnt} voxels shown, {pipe.map_occupied} occupied, {pipe.map_dropped} dropped")

    pipe.set_pose(0, gpu.mount_matrix(0, 0, 90, 1.0, 2.0, 0.0))
    cnt2 = pipe.build(live(now), map_on=False, min_count=1, neighbors="", radius=0.1, sensor=(0, 0, 0))
    c = pipe.c_xyz.numpy()[:cnt2].mean(0)
    pipe.set_pose(0, np.eye(4, dtype=np.float32))
    cnt3 = pipe.build(live(now), map_on=False, min_count=1, neighbors="", radius=0.1, sensor=(0, 0, 0))
    c0 = pipe.c_xyz.numpy()[:cnt3].mean(0)
    print(f"pose     : centroid {c0.round(2)} -> {c.round(2)} after yaw 90 deg + (1, 2, 0) offset")
    assert abs((c[0] - 1.0) + (c0[1])) < 0.05 and abs((c[1] - 2.0) - c0[0]) < 0.05


def replay(pipe, rec, out_pos, out_col, out_nrm):
    """A real recording through the Rust replay: ring, denoise, reflectivity range, map and export."""

    def v(now):
        return common.view(now, pipe, persist=0.0, voxel=0.02)

    pipe.clear_live()
    pipe.clear_map()
    src = ReplaySource(str(rec), speed=8.0, looped=False)
    now, total = 0.0, 0
    t_end = time.time() + 2.0
    while time.time() < t_end:
        time.sleep(1 / 60)
        b = src.poll()
        if b:
            xyz, attr, t, n = b
            now = float(t[:n].max())
            total += n
            pipe.ingest(xyz, attr, t, n, v(now), map_on=True, voxel=0.02)
    cnt = pipe.build(v(now), map_on=False, min_count=1, neighbors="count", radius=0.08, sensor=(0, 0, 0))
    pipe.shade(common.shade(now, gpu.MODE_REFLECTIVITY, denoise=1, min_nbrs=3), out_pos, out_col, out_nrm)
    lo_hi = pipe.scalar_percentiles()
    print(f"replay   : {total} pts ingested, now={now:.2f}s, {cnt} in ring, reflectivity p2..p98 = {lo_hi}")
    print("           ms", {k: round(v, 2) for k, v in pipe.ms.items()})
    cnt = pipe.build(v(now), map_on=True, min_count=2, neighbors="", radius=0.1, sensor=(0, 0, 0))
    print(f"replay map: {cnt} voxels (2 cm, >=2 hits) from {pipe.map_occupied} occupied")
    xyz, attr, _ = pipe.export()
    print("           bounds", xyz.min(0).round(2), xyz.max(0).round(2))


def main():
    dev = common.device()
    pipe = gpu.Pipeline(ring_capacity=1 << 22, map_capacity=1 << 22, device=dev)
    out_pos = wp.zeros(pipe.work_cap, dtype=wp.vec3, device=dev)
    out_col = wp.zeros(pipe.work_cap, dtype=wp.uint32, device=dev)
    out_nrm = wp.zeros(pipe.work_cap, dtype=wp.vec3, device=dev)
    print(f"GPU arrays: {pipe.gpu_bytes() / 2**20:.0f} MiB")
    simulated(pipe, dev, out_pos, out_col, out_nrm)
    rec = common.any_recording()
    if rec is None:
        print("replay   : skipped, no recording (set LIVOX_DATA_DIR / LIVOX_RECORDING)")
    else:
        replay(pipe, rec, out_pos, out_col, out_nrm)
    print("OK")
    sys.exit(0)


if __name__ == "__main__":
    main()
