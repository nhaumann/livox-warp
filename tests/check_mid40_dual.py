"""Synthetic dual-return LVXR2 file -> Rust replay -> NumPy -> Warp return filter (no hardware, no data)."""

import os
import struct
import sys
import tempfile
import time

import common
import numpy as np
from livox_warp import _native, gpu


def write_dual_recording(path):
    """Two packets of 100 dual-return points: return 0 and 1 of each firing share one timestamp."""
    with open(path, "wb") as f:
        f.write(b"LVXR2\n\x01\x01")
        for packet in range(2):
            header = bytes([5, 0, 1, 0, 0, 0, 0, 0, 0, 0]) + struct.pack("<Q", packet * 500_000)
            body = b"".join(struct.pack("<iiiB", 2000 + slot * 100, 0, 0, 200) for slot in range(100))
            data = header + body
            f.write(struct.pack("<QH", packet * 500_000, len(data)) + data)


def main():
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.abspath(os.path.join(directory, "dual.lvxr"))
        write_dual_recording(path)
        replay = _native.Replay(path, looped=False)
        deadline = time.monotonic() + 5
        while not replay.finished() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert replay.finished(), "replay did not finish"
        assert replay.stream_profile == "mid40-dual"
        xyz, attr, times, n = replay.drain()
        xyz = np.frombuffer(xyz, np.float32).reshape(-1, 3)
        attr = np.frombuffer(attr, np.uint32)
        times = np.frombuffer(times, np.float32)
        assert n == 200
        np.testing.assert_array_equal((attr >> 16) & 255, np.tile([0, 1], 100))
        np.testing.assert_array_equal(times[::2], times[1::2])
        np.testing.assert_allclose(times[100], 0.0005)
        assert replay.stats()["lost_packets"] == 0
        del replay

    pipe = gpu.Pipeline(ring_capacity=1024, map_capacity=1024, device=common.device())
    view = common.view(0.001, pipe, persist=0.0, voxel=0.03, max_range=100.0)
    pipe.set_pose(0, gpu.mount_matrix(0, 0, 0, 0, 0, 0))  # sensor = world (frame 0 of the pose table)
    pipe.ingest(xyz, attr, times, n, view, map_on=False, voxel=0.03)
    for mask, expected in [(3, 200), (1, 100), (2, 100)]:
        view.ret_mask = mask
        count = pipe.build(view, map_on=False, min_count=1, neighbors="", radius=0.1, sensor=(0, 0, 0))
        assert count == expected, (mask, count)
    print("OK: dual replay, shared firing timestamps, and first/second GPU filters")
    sys.exit(0)


if __name__ == "__main__":
    main()
