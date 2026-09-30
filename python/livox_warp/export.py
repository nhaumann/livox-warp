"""Point-cloud export of what the viewer shows: binary PLY, and LAS 1.4 with classification."""

from __future__ import annotations

import numpy as np


def write_ply(path: str, xyz: np.ndarray, rgba: np.ndarray | None, refl: np.ndarray):
    """Binary little-endian PLY with position, color and reflectivity (as 'intensity')."""
    n = len(xyz)
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if rgba is not None:
        fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    fields += [("intensity", "u1")]
    arr = np.empty(n, dtype=fields)
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if rgba is not None:
        arr["red"], arr["green"], arr["blue"] = rgba[:, 0], rgba[:, 1], rgba[:, 2]
    arr["intensity"] = refl
    props = "".join(
        f"property {'float' if t == '<f4' else 'uchar'} {name}\n" for name, t in fields
    )
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\n{props}end_header\n"
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        arr.tofile(f)


def write_las(path: str, xyz: np.ndarray, rgba: np.ndarray | None, refl: np.ndarray, ground: np.ndarray | None,
              t: np.ndarray | None = None):
    """LAS 1.4 point format 7 (xyz, intensity, RGB, GPS time, ASPRS classification: 2 ground, 1 other).

    Uncompressed: laspy writes LAZ only with a backend (lazrs) installed.
    """
    import laspy

    header = laspy.LasHeader(point_format=7, version="1.4")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.floor(xyz.min(axis=0)) if len(xyz) else np.zeros(3)
    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz[:, 0].astype(np.float64), xyz[:, 1].astype(np.float64), xyz[:, 2].astype(np.float64)
    las.intensity = refl.astype(np.uint16) * 257
    if rgba is not None:
        las.red = rgba[:, 0].astype(np.uint16) * 257
        las.green = rgba[:, 1].astype(np.uint16) * 257
        las.blue = rgba[:, 2].astype(np.uint16) * 257
    if ground is not None:
        las.classification = np.where(ground != 0, 2, 1).astype(np.uint8)
    else:
        las.classification = np.ones(len(xyz), np.uint8)
    if t is not None:
        las.gps_time = t.astype(np.float64)
    las.write(path)
