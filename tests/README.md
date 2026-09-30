# Tests, benchmarks and tools

Every script is a standalone program run from the repository root, e.g. `python tests/check_odometry.py`.
`python tests/run_all.py` runs every `check_*.py` and `smoke_*.py` and summarises OK / FAIL / SKIP with
timings (exit status 1 if anything failed); `python tests/run_all.py check_odometry smoke_pipeline` runs a
subset.

The simulated checks (`check_odometry`, `check_slam_worker`, `check_perception`, `check_mid40_dual`,
`smoke_pipeline`) need no data. The others use recordings and a prior map that you supply through the
environment; a script whose data is absent prints `SKIP: ...` and exits 0. `benchmarks/` holds the
scripts that measure rather than gate, `tools/` the calibration tool; both read the same data.

## Data layout

```
$LIVOX_DATA_DIR/                default: the repository root
    recordings/*.lvxr           raw packet recordings; a name containing "static" (the scanner standing
                                still) or "walk" (a handheld walk) is what the scripts pick by default
    maps/prior_20mm.npz         the prior map (a terrestrial scan, voxelised at 2 cm with normals)
    maps/recording_poses.npz    one 4x4 sensor-to-map pose per recording, keyed by the recording's file
                                name with dots replaced (`mid40_static.lvxr` -> `mid40_static_lvxr`)
    maps/walk_reference.npz     a frame-by-frame reference for the walk: `ref` (N, 4, 4) per 0.1 s frame,
                                `fit` (N,) the fraction of each frame's points within 3 cm of the scan, and
                                `trusted_until` (seconds) past which the reference itself is unreliable
```

How to create each:

- recordings: in the viewer (LiDAR > Recording) or with the CLI,
  `livox stream --lidar <lidar-ip> --secs 10 --record recordings/run.lvxr`
- the prior map: `python -m livox_warp.prior_map convert scan.e57 --voxel 0.02 --out maps/prior_20mm.npz`
- the poses: `python -m livox_warp.localize maps/prior_20mm.npz recordings/*static*.lvxr`
  (writes `maps/recording_poses.npz`; each recording must start with the scanner standing still inside
  the mapped area)
- the walk reference: written by whoever generates it (`ref`, `fit`, `trusted_until` as above)

## Environment

- `LIVOX_DATA_DIR`: the directory holding `recordings/` and `maps/` (default: the repository root)
- `LIVOX_PRIOR_MAP`: the prior map (default `$LIVOX_DATA_DIR/maps/prior_20mm.npz`)
- `LIVOX_STATIC_RECORDINGS`: recordings of the scanner standing still, joined with `os.pathsep`
  (default: every `*static*.lvxr` in `recordings/`)
- `LIVOX_WALK_RECORDING`: a handheld walk starting at rest in the mapped area (default: the first
  `*walk*.lvxr`)
- `LIVOX_RECORDING`: any recording, for the normals check, the neighbour benchmark and the smoke test
  (default: the first `*.lvxr`)
- `LIVOX_DEVICE`: the Warp device (default `cuda:0`; `cpu` runs the simulated checks without a GPU)
- `LIVOX_SIM_SCENE`: `check_odometry`'s room, `box` (bare, the default) or `office` (furnished)
- `LIVOX_TEST_SCALE`: `check_odometry`'s run-length multiplier (default `1.0`; lower on the CPU)
- `LIVOX_TEST_SPEEDUP`: `check_slam_worker`'s simulated seconds per wall second (default `2.0`)

## What each script needs

| script                          | data                                                                  |
|---------------------------------|-----------------------------------------------------------------------|
| `tests/check_real_static.py`    | static recordings                                                     |
| `tests/check_normals.py`        | any recording                                                         |
| `tests/check_perception.py`     | none; real-data part: a static recording, its pose and the prior map  |
| `tests/check_prior_map.py`      | the prior map and a static recording; step 3: the walk and reference  |
| `tests/smoke_pipeline.py`       | none; replay part: any recording                                      |
| `benchmarks/bench_neighbors.py` | any recording                                                         |
| `benchmarks/bench_prior_map.py` | the prior map and the walk (`--reference` for the comparison)         |
| `tools/calib_prior_map.py`      | the prior map and static recordings                                   |

`tests/common.py` resolves all of this (`static_recordings()`, `prior_map()`, `skip()`, ...) and holds the
helpers the scripts share: the `View` / `Shade` builders, recording replay to numpy, and the frame loop that
drives a `PriorSession` with the odometry worker the way the viewer does.
