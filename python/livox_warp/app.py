"""Livox Warp viewer: a pyglet window with imgui panels around the Warp GPU pipeline and the moderngl renderer.

Panels:
    LiDAR       discovered devices, the connected device (sampling, work mode, returns, recording, IP
                configuration, the mount pose stored on the LiDAR) and the other sources: the Warp
                simulator and LVXR replay.
    View        live window or integration map, colour modes, point rendering (size, EDL, surfels),
                denoise and filters, the viewer mount pose, grid and gizmo; fit, presets, export, screenshot.
    Perception  LiDAR-only odometry, ground segmentation, clusters and tracking, motion from map history,
                and the prior map (localisation in an earlier scan, drift correction, Changes colours).
    Stats       frame and GPU timings, packet counters, the point-rate history.

Command line: ``python -m livox_warp --help``. A source can be chosen up front (--lidar, --replay, --sim);
otherwise the viewer auto-connects to the first LiDAR it discovers. --frames with --screenshot or
--frames-dir renders a fixed number of frames and exits, for tests and README animations.

Config file: only the viewer mount pose is persisted (see Settings and --config); everything else starts
from the Settings defaults and the command line.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import threading
import time
import traceback
from dataclasses import dataclass, field

import numpy as np
import pyglet

pyglet.options["debug_gl"] = False

import imgui  # noqa: E402
import moderngl  # noqa: E402
import warp as wp  # noqa: E402
from imgui.integrations.pyglet import create_renderer  # noqa: E402

from . import _native, export, gpu  # noqa: E402
from .netcfg import add_host_alias, suggest_host  # noqa: E402
from .odom import Odometry  # noqa: E402
from .perception import Perception  # noqa: E402
from .prior_session import PriorSession  # noqa: E402
from .render import DrawOptions, OrbitCamera, Renderer, arrow_lines, box_lines  # noqa: E402
from .slam_worker import OdomWorker  # noqa: E402
from .sources import DEV_UNKNOWN, MID40, SIM_MOTIONS, SIM_SCENES, LiveSource, ReplaySource, SimSource  # noqa: E402

RETURN_MODES = ["Single first", "Single strongest", "Dual", "Triple"]
NOISE_LEVELS = ["normal", "noise: high", "noise: medium", "noise: low"]
COLOR_NAMES = [m.lower() for m in gpu.COLOR_MODES]  # --color choices, in gpu.MODE_* order
CIRCULAR_FOV = {MID40: 38.4}  # degrees, for the gizmo of sensors with a circular field of view
LOG_SLIDER = getattr(imgui, "SLIDER_FLAGS_LOGARITHMIC", 0)
RED, AMBER, GREEN, GREY = (1.0, 0.35, 0.3), (1.0, 0.75, 0.25), (0.45, 0.9, 0.5), (0.6, 0.62, 0.66)
PRIOR_LOG_COLORS = {"info": GREY, "ok": GREEN, "warn": AMBER}
PRIOR_HINT = "npz from: python -m livox_warp.prior_map convert scan.e57"
REPLAY_HINT = "path of an .lvxr file (Connected device > Recording writes them)"
# settings remembered between runs: only the viewer mount pose
CONFIG_PATH = os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~/.config"), "livox-warp", "viewer.json")


def load_config(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg if isinstance(cfg, dict) else {}
    except (OSError, ValueError):
        return {}


def save_config(path: str, cfg: dict) -> None:
    """Write atomically, so a crash mid-write never leaves a half file behind."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    os.replace(tmp, path)


@dataclass
class Settings:
    """Everything the panels edit. Only `mount` is persisted (in the config file); the rest starts from these
    defaults and the command line on every run."""

    map_mode: bool = False
    persist: float = 1.0
    persist_forever: bool = False
    voxel: float = 0.02
    map_min_count: int = 1
    color_mode: int = gpu.MODE_REFLECTIVITY
    color_lo: float = 0.0
    color_hi: float = 60.0
    auto_range: bool = True
    solid: tuple = (0.85, 0.9, 1.0)
    point_size: float = 2.0
    size_in_metres: bool = False
    world_size: float = 0.015
    round_points: bool = True
    edl: bool = True
    edl_strength: float = 0.7
    edl_radius: float = 1.5
    denoise: bool = False
    min_nbrs: int = 3
    radius: float = 0.10
    min_range: float = 0.1
    max_range: float = 500.0
    min_refl: int = 0
    noise_keep: list = field(default_factory=lambda: [True, True, True, True])
    ret_keep: list = field(default_factory=lambda: [True, True, True, True])
    crop_on: bool = False
    crop_lo: list = field(default_factory=lambda: [-20.0, -20.0, -5.0])
    crop_hi: list = field(default_factory=lambda: [20.0, 20.0, 5.0])
    mount: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # roll pitch yaw deg, x y z m
    grid: bool = True
    grid_z: float = -1.0
    gizmo: bool = True
    background: tuple = (0.055, 0.06, 0.075)
    freeze: bool = False
    # odometry (LiDAR-only)
    odom: bool = False
    frame_dt: float = 0.1
    map_voxel: float = 0.15  # registration map voxel (Odometry.reg_voxel)
    odom_iters: int = 12
    follow: bool = False
    show_path: bool = True
    # ground
    ground: bool = False
    gnd_cell: float = 0.5
    gnd_thresh: float = 0.2
    gnd_slope: float = 0.3
    gnd_thick: float = 0.15
    gnd_min_sup: int = 3
    gnd_use_normals: bool = True
    hide_ground: bool = False
    # clusters and tracking
    clusters: bool = False
    cl_voxel: float = 0.1
    cl_connect: float = 0.35
    cl_min_pts: int = 40
    cl_use_ground: bool = True
    boxes: bool = True
    labels: bool = True
    # motion from map history
    carve: bool = False
    carve_ratio: float = 2.0  # more rays through a surface than twice the returns on it
    carve_min: int = 3
    carve_stale: float = 1.0
    carve_hide: bool = True
    carve_every: int = 2
    # prior map (a scan of the building): Changes colours, the scan drawn for context
    prior_draw: bool = True
    prior_point_size: float = 1.5
    chg_near: float = 0.03
    chg_far: float = 0.10
    dyn_settle: float = 1.5
    hide_static: bool = False
    hide_dynamic: bool = False
    # surfels
    surfels: bool = False
    surfel_radius: float = 0.03


class App:
    def __init__(self, args):
        self.args = args
        w, h = args.size
        config = pyglet.gl.Config(double_buffer=True, depth_size=24, major_version=3, minor_version=3,
                                  forward_compatible=True)
        try:
            self.window = pyglet.window.Window(w, h, "Livox Warp", resizable=True, vsync=not args.no_vsync,
                                               config=config)
        except pyglet.window.NoSuchConfigException:
            self.window = pyglet.window.Window(w, h, "Livox Warp", resizable=True, vsync=not args.no_vsync)
        self.ctx = moderngl.create_context()

        imgui.create_context()
        self.imgui = create_renderer(self.window)
        imgui.get_io().font_global_scale = args.ui_scale
        imgui.style_colors_dark()

        wp.config.quiet = True
        wp.init()
        self.device = wp.get_device(args.device or ("cuda:0" if wp.is_cuda_available() else "cpu"))
        self.pipe = gpu.Pipeline(args.ring, args.map_slots, self.device)
        self.odom_worker = None  # created on first use: odometry runs on its own thread (slam_worker)
        self.odom_stats = Odometry.empty_stats()
        self.odom_traj = []  # mid-frame sensor positions, from the worker's results
        self.perc = Perception(self.pipe, self.device)
        self.gl = Renderer(self.ctx, self.pipe.work_cap, self.device)
        self.ev0 = wp.Event(self.device, enable_timing=True) if self.device.is_cuda else None
        self.ev1 = wp.Event(self.device, enable_timing=True) if self.device.is_cuda else None
        self.gpu_ms = 0.0

        self.cam = OrbitCamera()
        self.sensor_view = False  # Sensor camera view: follows the scanner's pose every frame
        self.s = Settings()
        if args.map_mode:
            self.s.map_mode = True
        if args.color:
            self.s.color_mode = COLOR_NAMES.index(args.color)  # argparse already checked it against COLOR_NAMES
        if args.persist is not None:
            self.s.persist = args.persist
        self.s.ground = bool(args.ground)
        self.s.clusters = bool(args.clusters)
        self.s.surfels = bool(args.surfels)
        self.s.carve = bool(args.carve)
        self.config_path = args.config or CONFIG_PATH
        self.config = load_config(self.config_path)
        mount = self.config.get("mount")
        if isinstance(mount, list) and len(mount) == 6 and all(isinstance(v, (int, float)) for v in mount):
            self.s.mount = [float(v) for v in mount]
        self.mount_saved = tuple(self.s.mount)
        self.mount_dirty_at = None  # when the pose last changed and is not yet saved
        self._mount_pending = None
        self.source = None
        self.now = 0.0
        self.last_t_seen = None
        self.dev_type = DEV_UNKNOWN
        self.voxel_used = self.s.voxel
        self.mount_used = None
        self.mount = gpu.mount_matrix(*self.s.mount[:3], *self.s.mount[3:])
        self.pose_now = self.mount.astype(np.float64)
        self.tracks = []
        self.label_items = []
        self.traj_len = -1
        self._traj_seg = None
        self._odom_cfg = None
        self.pipe.set_pose(0, self.mount, None, self.pipe.epoch)
        self.prior = None  # PriorSession: localisation in a prior map, off the render thread
        self.prior_path = args.prior_map or ""
        self._prior_drawn = False

        self.logs = collections.deque(maxlen=12)
        self.pending = collections.deque()
        self.busy = None
        self.rate_hist = np.zeros(120, np.float32)
        self.rate_last = (time.perf_counter(), 0)
        self.pps = 0.0
        self.fps = 0.0
        self.frame_ms = 0.0
        self.last_range_update = 0.0
        self.status = None
        self.status_time = 0.0
        self.stats = {}
        self.screenshot_req = None
        self.frames = 0
        self.frames_saved = 0  # frames written to --frames-dir
        if args.frames_dir:
            os.makedirs(args.frames_dir, exist_ok=True)
        self.fit_at = None

        # device-panel inputs
        self.host_choice = {}
        self.adapter_idx = 0
        self.ip_dynamic = False
        self.ip_edit = ["", "", ""]  # ip, mask, gateway: filled from the device or suggested when the form opens
        self.ip_confirm = False
        self.ip_current = None
        self.ret_mode = 0
        self.dev_extrinsic = None
        self.replay_path = args.replay or ""
        self.replay_speed = 1.0
        self.record_path = ""
        self.sim_motion = SIM_MOTIONS.index(args.sim_motion)
        self.sim_scene = SIM_SCENES.index(args.sim_scene)
        self.ifaces = _native.local_interfaces()

        try:
            self.discovery = _native.Discovery()
        except RuntimeError as e:
            self.discovery = None
            self.log(f"discovery unavailable: {e}", RED)
        self.discovered = []
        self.disc_time = 0.0
        self.auto_connect = not args.no_auto and not (args.sim or args.replay or args.lidar)
        self.auto_deadline = time.perf_counter() + 4.0

        self.window.push_handlers(self)
        self.gl.set_grid(self.s.grid_z)
        self._update_gizmo()

        if not self.gl.interop:
            self.log("CPU FALLBACK: CUDA-GL interop unavailable; Warp copies through host memory", RED)
        self.log(f"Warp {wp.config.version} on {self.device.name}; "
                 f"{self.pipe.gpu_bytes() / 2**20:.0f} MiB of GPU buffers", GREY)
        if any(self.s.mount):
            r, p, y, x, yy, z = self.s.mount
            self.log(f"mount pose from {self.config_path}: roll {r:g} pitch {p:g} yaw {y:g} deg, "
                     f"xyz {x:g} {yy:g} {z:g} m", GREY)

        if args.sim:
            self.set_source(SimSource(device=self.device, motion=args.sim_motion, scene=args.sim_scene))
        elif args.replay:
            self.open_replay(args.replay, args.speed)
        elif args.lidar:
            host = args.host or _native.host_for(args.lidar)
            if host is None:
                self.log(f"no local address on {args.lidar}'s subnet; pass --host or add one", RED)
            else:
                self.connect(args.lidar, host, args.type)
        if args.odom:
            self.set_odom(True)
        if args.prior_map:
            self.load_prior(args.prior_map)

    # ---- plumbing -------------------------------------------------------------------------

    def log(self, text: str, color=GREY):
        self.logs.append((time.strftime("%H:%M:%S"), text, color))
        print(text, flush=True)

    def _prior_log(self, text: str, level: str):
        """PriorSession's log callback: its info / ok / warn levels in the viewer's colours."""
        self.log(text, PRIOR_LOG_COLORS.get(level, GREY))

    def job(self, label: str, fn, on_ok=None):
        """Run a blocking device call off the UI thread; callbacks run back on the UI thread."""
        if self.busy:
            self.log(f"busy with {self.busy}", AMBER)
            return

        def run():
            try:
                r = fn()
                self.pending.append(lambda: self._job_ok(label, r, on_ok))
            except Exception as e:  # noqa: BLE001 - surface every device error in the log
                msg = f"{label} failed: {e}"  # bind now: `e` is gone once the except block ends
                self.pending.append(lambda: self.log(msg, RED))
            finally:
                self.busy = None

        self.busy = label
        threading.Thread(target=run, daemon=True).start()

    def _job_ok(self, label, result, on_ok):
        if on_ok:
            on_ok(result)
        else:
            self.log(f"{label}: done", GREEN)

    def set_source(self, src):
        if self.source is not None:
            try:
                self.source.close()
            except Exception as e:  # noqa: BLE001
                self.log(f"closing source: {e}", AMBER)
        self.source = src
        self.clear_all()
        if self.prior is not None:
            self.prior.forget()  # another source: another place
            self.prior.clock_restarted()
        self.status = None
        self.ip_current = None
        self.ip_edit = ["", "", ""]
        self.dev_extrinsic = None
        self.fit_at = time.perf_counter() + 1.0  # frame the cloud once a second of data is in
        if src is not None:
            self.dev_type = src.dev_type
            self.log(f"source: {src.label}", GREEN)
            self._update_gizmo()

    def clear_all(self):
        """Forget points, map, odometry and tracks; the next frame starts at the mount pose."""
        if self.prior is not None:
            # the world restarts at the mount pose with the sensor where it is: keep its place in the map
            mount = gpu.mount_matrix(*self.s.mount[:3], *self.s.mount[3:]).astype(np.float64)
            self.prior.world_reset(self.pose_now, mount)
        self.pipe.clear_live()
        self.pipe.clear_map()
        self.now = 0.0
        self.last_t_seen = None
        if self.odom_worker is not None:
            self.odom_worker.reset(self.mount.astype(np.float64))
        self.odom_stats = Odometry.empty_stats()
        self.odom_traj = []
        self._traj_seg = None
        self.pipe.epoch += 1  # pose-table entries from before the clear no longer count
        self.pipe.set_pose(0, self.mount, None, self.pipe.epoch)
        self.pose_now = self.mount.astype(np.float64)
        self.perc.tracker.reset()
        self.tracks = []
        self.traj_len = -1
        self._update_gizmo()

    def set_odom(self, on: bool):
        self.s.odom = on
        if on and self.odom_worker is None:
            self.odom_worker = OdomWorker(self.pipe, self.device)
            self._configure_odom()
        if on:
            # points are now drawn in the world frame and leave a fixed view as the sensor moves: follow it
            self.s.follow = True
        self.clear_all()
        self.log("odometry on: frames register against the map; world = first frame's mount pose" if on
                 else "odometry off", GREY)

    def connect(self, lidar_ip: str, host_ip: str, dev_type: int):
        if isinstance(self.source, LiveSource):
            self.set_source(None)
        self.job(
            f"connect {lidar_ip}",
            lambda: LiveSource(lidar_ip, host_ip, dev_type),
            on_ok=self.set_source,
        )

    def open_replay(self, path: str, speed: float):
        try:
            self.set_source(ReplaySource(path, speed, True))
        except Exception as e:  # noqa: BLE001 - a bad path or file must not take the viewer down
            self.log(f"replay {path}: {e}", RED)

    def act(self, label: str, fn):
        """Run a UI action that may raise (files, devices) and log a failure instead of letting it
        escape an open imgui window."""
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self.log(f"{label} failed: {e}", RED)
            return None

    def _update_gizmo(self):
        self.gl.set_gizmo(self.pose_now.astype(np.float32), CIRCULAR_FOV.get(self.dev_type))

    def make_view(self) -> gpu.View:
        s = self.s
        v = gpu.View()
        v.now = self.now
        v.persist = 0.0 if s.persist_forever else s.persist
        v.min_range, v.max_range, v.min_refl = s.min_range, s.max_range, s.min_refl
        v.noise_mask = sum(1 << k for k, on in enumerate(s.noise_keep) if on)
        v.ret_mask = sum(1 << k for k, on in enumerate(s.ret_keep) if on)
        v.crop_on = 1 if s.crop_on else 0
        v.crop_lo, v.crop_hi = wp.vec3(*s.crop_lo), wp.vec3(*s.crop_hi)
        v.inv_frame_dt = 1.0 / s.frame_dt if s.odom else 0.0
        v.epoch = self.pipe.epoch
        v.inv_voxel = 1.0 / max(s.voxel, 1e-4)
        v.map_mask = self.pipe.map_cap - 1
        v.dyn_on = 1 if self.want_dyn() else 0
        v.dyn_settle = s.dyn_settle
        v.carve_on = 1 if (s.carve and s.carve_hide) else 0
        v.carve_ratio = s.carve_ratio
        v.carve_min = s.carve_min
        v.carve_stale = s.carve_stale
        return v

    def want_dyn(self) -> bool:
        s = self.s
        return s.color_mode == gpu.MODE_MOTION or s.hide_static or s.hide_dynamic

    def make_shade(self, has_normals: bool) -> gpu.Shade:
        s = self.s
        sh = gpu.Shade()
        sh.mode = s.color_mode
        sh.lo, sh.hi = s.color_lo, s.color_hi
        sh.solid = wp.vec3(*s.solid)
        sh.now = self.now
        sh.sensor = wp.vec3(*self.pose_now[:3, 3])
        sh.eye = wp.vec3(*self.cam.eye)
        sh.denoise = 1 if s.denoise else 0
        sh.min_nbrs = s.min_nbrs
        sh.has_normals = 1 if has_normals else 0
        sh.has_gnd = 1 if s.ground else 0
        sh.has_cid = 1 if s.clusters else 0
        sh.has_dyn = 1 if self.want_dyn() else 0
        sh.hide_gnd = 1 if (s.ground and s.hide_ground) else 0
        sh.hide_dyn = 1 if s.hide_dynamic else 0
        sh.hide_static = 1 if s.hide_static else 0
        sh.has_chg = 1 if (s.color_mode == gpu.MODE_CHANGES and self.prior is not None and self.prior.localised) else 0
        sh.chg_near, sh.chg_far = s.chg_near, s.chg_far
        return sh

    # ---- per frame -----------------------------------------------------------------------

    def ingest(self):
        if self.source is None or self.s.freeze:
            return
        b = self.source.poll()
        if b is None:
            return
        xyz, attr, t, n = b
        t_min, t_max = float(self.source.last_t0), float(self.source.last_t1)
        if self.last_t_seen is not None and t_max < self.last_t_seen - 2.0:
            # Replay looped or the LiDAR clock restarted: start the live window over, move the map's
            # timestamps onto the new clock, and stop trusting pose-table entries of the old one.
            self.pipe.rebase_clock(self.now - t_min)
            self.pipe.clear_live()
            self.now = 0.0
            self.pipe.epoch += 1
            self.pipe.set_pose(0, self.mount, None, self.pipe.epoch)
            if self.s.odom and self.odom_worker is not None:
                self.odom_worker.restart_clock()
            if self.prior is not None:
                self.prior.clock_restarted()
        self.last_t_seen = t_max
        if self.prior is not None and isinstance(xyz, np.ndarray):
            self.prior.add_points(xyz, attr, t, n)
        self.now = max(self.now, t_max) if self.now else t_max
        s = self.s
        view = self.make_view()
        if s.odom and self.odom_worker is not None:
            sx, sa, st, sf = self.pipe.stage(xyz, attr, t, n, 1.0 / s.frame_dt)
            self.pipe.ingest_ring(sx, sa, st, sf, n)
            # owned snapshots; dropped (counted) if the worker is behind
            self.odom_worker.submit(sx, sa, st, n, t_min, t_max, self.pipe.epoch)
        else:
            sx, sa, st, sf = self.pipe.stage(xyz, attr, t, n, 0.0)
            self.pipe.ingest_ring(sx, sa, st, sf, n)
            self.pipe.map_insert(sx, sa, st, sf, n, view, s.voxel)
            if s.carve:
                self.pipe.carve(sx, sa, sf, n, s.voxel, every=s.carve_every)

    def _configure_odom(self):
        s = self.s
        self.odom_worker.configure(frame_dt=s.frame_dt, reg_voxel=s.map_voxel, max_iter=s.odom_iters)
        self._odom_cfg = (s.frame_dt, s.map_voxel, s.odom_iters)

    def _consume_odometry(self):
        """Frames the worker finished: bake their deskewed points into the map, carve, move the sensor."""
        if self.odom_worker is None or not self.s.odom:
            return
        s = self.s
        results = self.odom_worker.poll()
        if not results:
            return
        view = self.make_view()
        for r in results:
            if r.snaps:
                fx, fa, ft, ff = r.arrays()
                self.pipe.map_insert(fx, fa, ft, ff, r.m, view, s.voxel)
                if s.carve:
                    self.pipe.carve(fx, fa, ff, r.m, s.voxel, every=s.carve_every)
            self.odom_traj.append(r.T[:3, 3].astype(np.float32))
            self.odom_stats = r.stats
            if self.prior is not None:
                self.prior.add_frame(r.k, r.T, r.xi)
                st = r.stats
                self.prior.odom_note = (f"speed {st.get('speed', 0.0):.2f} m/s, turn {st.get('turn', 0.0):.0f} deg/s, "
                                        f"{st.get('degen', 0)} held directions, {st.get('corr', 0)} correspondences, "
                                        f"rms {st.get('rms', 0.0) * 100:.1f} cm, {st.get('gaps', 0)} gaps")
        T = results[-1].T
        self.pose_now = T
        self._update_gizmo()
        if s.follow and not self.sensor_view:
            # centre the view on what the sensor sees: a few metres ahead along its forward (x) axis
            self.cam.target = T[:3, 3] + 3.0 * T[:3, 0]

    def frame(self):
        s = self.s
        while self.pending:
            self.pending.popleft()()

        fb_w, fb_h = self.window.get_framebuffer_size()
        self.imgui.process_inputs()
        imgui.new_frame()
        self.ui()

        # settings that invalidate the map (it lives in world space at a fixed voxel size)
        mount_key = tuple(s.mount)
        if mount_key != self.mount_used:
            self.mount = gpu.mount_matrix(*s.mount[:3], *s.mount[3:])
            if self.mount_used is not None:
                self.clear_all()
            else:
                self.pipe.set_pose(0, self.mount, None, self.pipe.epoch)
                self.pose_now = self.mount.astype(np.float64)
                self._update_gizmo()
            self.mount_used = mount_key
        if mount_key != self.mount_saved:
            # typing or dragging changes the pose every frame: save once it has settled for half a second
            if self.mount_dirty_at is None or self._mount_pending != mount_key:
                self.mount_dirty_at, self._mount_pending = time.perf_counter(), mount_key
            elif time.perf_counter() - self.mount_dirty_at > 0.5:
                self._save_mount()
        if s.voxel != self.voxel_used:
            self.pipe.clear_map()
            self.voxel_used = s.voxel
        if self.odom_worker is not None and self._odom_cfg != (s.frame_dt, s.map_voxel, s.odom_iters):
            restart = self._odom_cfg[:2] != (s.frame_dt, s.map_voxel)
            self._configure_odom()
            if restart and s.odom:
                self.clear_all()

        if self.ev0 is not None:
            wp.record_event(self.ev0)
        self.ingest()
        self._consume_odometry()
        if self.sensor_view:
            # the Sensor view is a mode: the eye stays at the scanner, looking along it, as the pose moves
            T = self.pose_now
            self.cam.look_from(T[:3, 3], T[:3, 0], 3.0)
        if self.prior is not None:
            self.prior.frame_dt = s.frame_dt
            self.prior.tick(time.perf_counter(), self.pose_now, s.odom)
            if self.prior.ready and not self._prior_drawn:
                draw = self.prior.draw
                if draw is not None:
                    self.gl.set_prior(*draw)
                    self._prior_drawn = True
        mode = s.color_mode
        need_normals = mode in gpu.NEEDS_NORMALS or s.surfels or (s.ground and s.gnd_use_normals)
        neighbors = "normals" if need_normals else ("count" if s.denoise else "")
        count = self.pipe.build(self.make_view(), s.map_mode, s.map_min_count, neighbors, s.radius,
                                self.pose_now[:3, 3])
        if self.odom_worker is not None:
            # build() synchronized the render stream past this frame's map inserts: the slots may be reused
            self.odom_worker.release_consumed()
        if s.ground:
            self.perc.ground(cell=s.gnd_cell, thick=s.gnd_thick, min_sup=s.gnd_min_sup, thresh=s.gnd_thresh,
                             slope_step=s.gnd_slope, use_normals=s.gnd_use_normals, nz_min=0.7)
        if s.clusters:
            self.perc.cluster(s.cl_voxel, s.cl_connect, s.cl_min_pts, s.cl_use_ground and s.ground)
            self.tracks = self.perc.track(self.now)
        else:
            self.tracks = []
        if mode == gpu.MODE_CHANGES and self.prior is not None and self.prior.localised and self.prior.grid is not None:
            # distance of every visible point to the scan (read-only on the grid: safe on the render stream)
            self.prior.grid.distances(self.pipe.c_xyz, count, self.prior.T_MW, self.pipe.chg, None)
        pos, col, nrm = self.gl.map()
        self.pipe.shade(self.make_shade(need_normals), pos, col, nrm)
        if self.ev1 is not None:
            wp.record_event(self.ev1)
        self.gl.unmap()
        if self.ev0 is not None:
            self.gpu_ms = wp.get_event_elapsed_time(self.ev0, self.ev1)

        tnow = time.perf_counter()
        if self.fit_at is not None and tnow >= self.fit_at and count > 1000:
            self.fit_view()
            z = self.pipe.floor_z()
            if z is not None:
                s.grid_z = round(z, 2)
            if self.args.view:
                self.camera_preset(self.args.view)
            self.fit_at = None
        if s.auto_range and mode in gpu.SCALAR_MODES and tnow - self.last_range_update > 0.5 and count:
            r = self.pipe.scalar_percentiles()
            if r:
                lo, hi = r
                if mode in (gpu.MODE_REFLECTIVITY, gpu.MODE_LIT, gpu.MODE_GROUND, gpu.MODE_SPEED):
                    lo = 0.0  # reflectivity, height above ground and speed read best anchored at zero
                if hi - lo < 1e-3:
                    hi = lo + 1.0
                s.color_lo, s.color_hi = lo, hi
            self.last_range_update = tnow

        self._update_overlay()
        self.gl.set_grid(s.grid_z)
        prior_draw = None
        if self.prior is not None and self.prior.localised and s.prior_draw:
            prior_draw = (np.linalg.inv(self.prior.T_MW), s.prior_point_size)
        self.gl.draw(count, self.cam, (fb_w, fb_h), DrawOptions(
            background=s.background, grid=s.grid, gizmo=s.gizmo, point_size=s.point_size,
            size_in_metres=s.size_in_metres, world_size=s.world_size, round_points=s.round_points,
            edl=s.edl, edl_strength=s.edl_strength, edl_radius=s.edl_radius,
            surfels=s.surfels and need_normals, surfel_radius=s.surfel_radius, prior=prior_draw))
        imgui.render()
        self.imgui.render(imgui.get_draw_data())

        if self.screenshot_req:
            self._save_screenshot(self.screenshot_req, fb_w, fb_h)
            self.screenshot_req = None
        if self.args.frames_dir and self.frames % self.args.frames_every == 0:
            # a numbered frame sequence (frame_000001.png, ...) that a GIF assembler can pick up offline
            self.frames_saved += 1
            path = os.path.join(self.args.frames_dir, f"frame_{self.frames_saved:06d}.png")
            self._save_screenshot(path, fb_w, fb_h, announce=False)

    def _update_overlay(self):
        """Trajectory, track boxes and velocity arrows as one line list; labels for the next UI pass."""
        s = self.s
        verts = []
        self.label_items = []
        if s.odom and s.show_path and len(self.odom_traj) >= 2:
            seg = self._traj_seg
            n = len(self.odom_traj)
            if seg is None or seg[0] != n:
                # extend the cached segment array with the new poses only (the path grows by ~10/s)
                old = seg[1] if seg is not None else np.zeros((0, 6), np.float32)
                start = max(1, old.shape[0] // 2 + 1)
                pts = np.array(self.odom_traj[start - 1:], dtype=np.float32).reshape(-1, 3)
                add = np.empty((2 * (len(pts) - 1), 6), np.float32)
                add[0::2, :3] = pts[:-1]
                add[1::2, :3] = pts[1:]
                add[:, 3:] = (0.35, 0.95, 0.85)
                self._traj_seg = seg = (n, np.concatenate([old, add]))
            verts.append(seg[1])
        if s.clusters and (s.boxes or s.labels):
            rows = []
            for tid, tr in self.tracks:
                col = tuple(_hue(tid))
                speed = float(np.linalg.norm(tr["vel"]))
                if s.boxes:
                    rows += box_lines(tr["lo"], tr["hi"], col)
                    if speed > 0.15:
                        rows += arrow_lines(tr["pos"], tr["vel"], (1.0, 0.9, 0.3))
                if s.labels:
                    top = np.array([tr["pos"][0], tr["pos"][1], tr["hi"][2]])
                    self.label_items.append((top, f"#{tid} {speed:.1f} m/s", col))
            if rows:
                verts.append(np.array(rows, np.float32))
        self.gl.set_overlay(np.concatenate(verts) if verts else None)

    def fit_view(self):
        b = self.pipe.bounds()
        if b is not None:
            self.cam.fit(*b)

    def _save_screenshot(self, path, w, h, announce: bool = True):
        from PIL import Image

        data = self.ctx.screen.read(viewport=(0, 0, w, h), components=3)
        _ = self.ctx.error  # the read leaves a GL error behind that would fail the next frame's imgui pass
        Image.frombytes("RGB", (w, h), data).transpose(Image.FLIP_TOP_BOTTOM).save(path)
        if announce:
            self.log(f"screenshot: {path}", GREEN)

    def _update_rates(self):
        now = time.perf_counter()
        if self.source is not None:
            if now - self.status_time > 0.25:
                self.stats = self.source.stats() or {}
                self.status = self.source.status()
                self.status_time = now
            t_last, p_last = self.rate_last
            if now - t_last >= 1.0:
                pts = self.stats.get("points", 0)
                self.pps = (pts - p_last) / (now - t_last) if p_last else 0.0
                self.rate_hist = np.roll(self.rate_hist, -1)
                self.rate_hist[-1] = self.pps
                self.rate_last = (now, pts)
        if self.discovery is not None and now - self.disc_time > 0.5:
            self.discovered = self.discovery.devices(3.0)
            self.disc_time = now
            if self.auto_connect and not self.busy and self.source is None:
                reachable = [d for d in self.discovered if d["host"]]
                if reachable:
                    d = reachable[0]
                    self.auto_connect = False
                    self.log(f"auto-connecting to {d['type_name']} {d['ip']}", GREY)
                    self.connect(d["ip"], d["host"], d["type"])
                elif now > self.auto_deadline:
                    self.auto_connect = False

    # ---- UI ------------------------------------------------------------------------------

    def ui(self):
        self._update_rates()
        self.ui_labels()
        self.ui_device()
        self.ui_view()
        self.ui_perception()
        self.ui_stats()

    def ui_labels(self):
        """Track labels drawn over the scene (positions from the last rendered frame)."""
        if not self.label_items:
            return
        w, h = self.window.get_size()
        pts = np.array([p for p, _, _ in self.label_items])
        xs, ys, ok = self.cam.project(pts, (w, h))
        dl = imgui.get_background_draw_list()
        for (p, text, col), x, y, vis in zip(self.label_items, xs, ys, ok):
            if vis:
                dl.add_text(float(x) + 6.0, float(y) - 8.0, imgui.get_color_u32_rgba(col[0], col[1], col[2], 1.0), text)

    def ui_device(self):
        imgui.set_next_window_position(10, 10, imgui.FIRST_USE_EVER)
        imgui.set_next_window_size(390, 640, imgui.FIRST_USE_EVER)
        imgui.begin("LiDAR")
        if self.busy:
            imgui.text_colored(f"working: {self.busy} ...", *AMBER)

        if imgui.collapsing_header("Discovered", flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]:
            if self.discovery is None:
                imgui.text_colored("UDP 55000 unavailable", *RED)
            elif not self.discovered:
                imgui.text_colored("listening for broadcasts on UDP 55000 ...", *GREY)
                if self.discovery.datagrams() == 0:
                    imgui.text_wrapped("Nothing heard yet. A connected LiDAR stops broadcasting; "
                                       "Livox Viewer also hides broadcasts while it is open.")
            for d in self.discovered:
                self.ui_discovered(d)

        live = self.source if isinstance(self.source, LiveSource) else None
        if live is not None and imgui.collapsing_header("Connected device", flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]:
            self.ui_live(live)

        if imgui.collapsing_header("Other sources")[0]:
            _, self.sim_motion = imgui.combo("sim motion", self.sim_motion, SIM_MOTIONS)
            _, self.sim_scene = imgui.combo("sim scene", self.sim_scene, SIM_SCENES)
            if imgui.button("Simulated Mid-40 (Warp ray-cast)"):
                self.act("simulator", lambda: self.set_source(SimSource(
                    device=self.device, motion=SIM_MOTIONS[self.sim_motion], scene=SIM_SCENES[self.sim_scene])))
            _, self.replay_path = imgui.input_text("file", self.replay_path, 512)
            if not self.replay_path:
                imgui.text_colored(REPLAY_HINT, *GREY)
            _, self.replay_speed = imgui.slider_float("speed", self.replay_speed, 0.1, 8.0, "%.1fx")
            if imgui.button("Open replay"):
                if self.replay_path:
                    self.open_replay(self.replay_path, self.replay_speed)
                else:
                    self.log("enter the path of an .lvxr recording first", AMBER)
            if self.source is not None:
                imgui.same_line()
                if imgui.button("Close source"):
                    self.set_source(None)

        imgui.separator()
        for stamp, text, color in list(self.logs)[-7:]:
            imgui.text_colored(f"{stamp} {text}", *color)
        imgui.end()

    def ui_discovered(self, d):
        imgui.push_id(d["code"])
        imgui.text(f"{d['type_name']}  {d['ip']}")
        imgui.same_line()
        imgui.text_colored(d["code"], *GREY)
        local = [ip for _, ip, _ in self.ifaces]
        options = ["auto"] + local
        idx = self.host_choice.get(d["code"], 0)
        _, idx = imgui.combo("host ip", idx, options)
        self.host_choice[d["code"]] = idx
        host = d["host"] if idx == 0 else options[idx]
        if host:
            if imgui.button("Connect"):
                self.connect(d["ip"], host, d["type"])
            imgui.same_line()
            imgui.text_colored(f"stream to {host}", *GREEN)
        else:
            imgui.text_colored("No local address on this LiDAR's subnet.", *RED)
            names = [n for n, _, _ in self.ifaces] or ["Ethernet"]
            self.adapter_idx = min(self.adapter_idx, len(names) - 1)
            _, self.adapter_idx = imgui.combo("adapter", self.adapter_idx, names)
            alias = suggest_host(d["ip"])
            if imgui.button(f"Add {alias} to {names[self.adapter_idx]} (admin prompt)"):
                self.job("add host address", lambda: add_host_alias(names[self.adapter_idx], alias),
                         on_ok=lambda _: self._refresh_ifaces())
        imgui.separator()
        imgui.pop_id()

    def _refresh_ifaces(self):
        self.ifaces = _native.local_interfaces()
        self.log("host addresses refreshed", GREEN)

    def ui_live(self, live: LiveSource):
        st = self.status or {}
        ok = st.get("connected", False)
        imgui.text(f"{st.get('type_name', '?')}  {st.get('lidar_ip', '')}  ->  {st.get('host_ip', '')}")
        imgui.text_colored(f"{'connected' if ok else 'HEARTBEAT LOST'}   state: {st.get('state_name') or '?'}   "
                           f"fw {st.get('firmware') or '?'}", *(GREEN if ok else RED))
        problems = st.get("problems") or []
        imgui.text_colored("health: " + (", ".join(problems) if problems else "ok"), *(AMBER if problems else GREEN))
        age = st.get("heartbeat_age")
        imgui.text_colored(f"heartbeat {age:.1f}s ago" if age is not None else "no heartbeat yet", *GREY)

        if live.sampling:
            if imgui.button("Stop sampling"):
                self.job("stop sampling", live.stop)
        else:
            if imgui.button("Start sampling"):
                self.job("start sampling", live.start)
        imgui.same_line()
        if imgui.button("Disconnect"):
            self.set_source(None)

        imgui.text("Work mode")
        for name, m in (("Normal", 1), ("Power-save", 2), ("Standby", 3)):
            imgui.same_line()
            if imgui.button(name):
                self.job(f"mode {name}", lambda m=m: live.dev.set_mode(m))

        if self.dev_type != MID40:
            changed, self.ret_mode = imgui.combo("returns", self.ret_mode, RETURN_MODES)
            if changed:
                self.job("return mode", lambda m=self.ret_mode: live.dev.set_return_mode(m))
            for label, fn in (("Fan on", lambda: live.dev.set_fan(True)), ("Fan off", lambda: live.dev.set_fan(False)),
                              ("Rain/fog on", lambda: live.dev.set_rain_fog(True)),
                              ("Rain/fog off", lambda: live.dev.set_rain_fog(False))):
                if imgui.button(label):
                    self.job(label, fn)
                imgui.same_line()
            imgui.new_line()
            if imgui.button("IMU 200 Hz"):
                self.job("imu on", lambda: live.dev.set_imu(True))
            imgui.same_line()
            if imgui.button("IMU off"):
                self.job("imu off", lambda: live.dev.set_imu(False))
            imu = live.dev.imu()
            if imu:
                (gx, gy, gz), (ax, ay, az) = imu
                imgui.text_colored(f"gyro {gx:+.3f} {gy:+.3f} {gz:+.3f}  acc {ax:+.2f} {ay:+.2f} {az:+.2f}", *GREY)
        else:
            if st.get("stream_profile") == "mid40-dual":
                imgui.text_colored("Mid-40: dual return, 100 kHz firings", *GREEN)
                imgui.text_wrapped("Second-return intensity is a fixed marker (200), not measured reflectivity. "
                                   "Rain/fog suppression is unavailable.")
            else:
                imgui.text_colored("Mid-40: single-return stream", *GREY)

        if imgui.tree_node("Recording"):
            self._ui_recording(live)
            imgui.tree_pop()
        if imgui.tree_node("IP configuration"):
            self._ui_ip_config(live)
            imgui.tree_pop()
        if imgui.tree_node("Mount pose stored on the LiDAR"):
            self._ui_extrinsic(live)
            imgui.tree_pop()

    def _ui_recording(self, live: LiveSource):
        if not self.record_path:
            self.record_path = os.path.join("recordings", time.strftime("lidar_%Y%m%d_%H%M%S.lvxr"))
        _, self.record_path = imgui.input_text("path", self.record_path, 512)
        if live.recording:
            imgui.text_colored(f"recording to {live.recording}", *RED)
            if imgui.button("Stop recording"):
                path = live.recording
                n = self.act("stop recording", live.stop_recording)
                if n is not None:  # the source keeps its recording state if stopping failed
                    self.log(f"recorded {n} packets to {path}", GREEN)
                    self.record_path = ""
        elif imgui.button("Start recording (raw packets, replayable)"):
            path = self.record_path

            def start():
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                live.start_recording(path)

            self.act("start recording", start)

    def _ui_ip_config(self, live: LiveSource):
        if not any(self.ip_edit):
            # nothing read from the device yet: suggest a static address on the LiDAR's own subnet
            subnet = suggest_host(live.lidar_ip).rsplit(".", 1)[0]
            self.ip_edit = [live.lidar_ip, "255.255.255.0", f"{subnet}.1"]
        if imgui.button("Read from LiDAR"):
            self.job("read ip", live.dev.ip_info, on_ok=self._got_ip)
        if self.ip_current:
            c = self.ip_current
            mode = "dynamic (DHCP)" if c["dynamic"] else "static"
            imgui.text(f"{mode}  {c['ip']}  mask {c['mask']}  gw {c['gateway']}")
        if imgui.radio_button("static", not self.ip_dynamic):
            self.ip_dynamic = False
        imgui.same_line()
        if imgui.radio_button("dynamic (DHCP)", self.ip_dynamic):
            self.ip_dynamic = True
        if not self.ip_dynamic:
            for k, label in enumerate(("ip", "mask", "gateway")):
                _, self.ip_edit[k] = imgui.input_text(label, self.ip_edit[k], 32)
        _, self.ip_confirm = imgui.checkbox("I understand the LiDAR reboots and moves", self.ip_confirm)
        if self.ip_confirm and imgui.button("Apply and reboot"):
            ip, mask, gw = self.ip_edit
            dyn = self.ip_dynamic

            def apply():
                live.dev.set_ip(dyn, ip, mask, gw)
                live.dev.reboot(200)

            self.ip_confirm = False
            self.job("set ip + reboot", apply,
                     on_ok=lambda _: (self.log("LiDAR rebooting; reconnect from Discovered in ~15 s", GREEN),
                                      self.set_source(None)))

    def _ui_extrinsic(self, live: LiveSource):
        if imgui.button("Read"):
            self.job("read extrinsic", live.dev.extrinsic, on_ok=self._got_extrinsic)
        if self.dev_extrinsic:
            r, p, y, x, yy, z = self.dev_extrinsic
            imgui.text(f"roll {r:.2f} pitch {p:.2f} yaw {y:.2f} deg; xyz {x} {yy} {z} mm")
            if imgui.button("Use as viewer mount pose"):
                self.s.mount = [r, p, y, x / 1000.0, yy / 1000.0, z / 1000.0]
        if imgui.button("Write viewer mount pose to LiDAR"):
            m = self.s.mount
            self.job("write extrinsic", lambda: live.dev.set_extrinsic(
                m[0], m[1], m[2], int(round(m[3] * 1000)), int(round(m[4] * 1000)), int(round(m[5] * 1000))))

    def _got_ip(self, info):
        self.ip_current = info
        self.ip_dynamic = info["dynamic"]
        if not info["dynamic"]:
            self.ip_edit = [info["ip"], info["mask"], info["gateway"]]
        self.log(f"LiDAR ip: {'dynamic' if info['dynamic'] else 'static'} {info['ip']}", GREEN)

    def _got_extrinsic(self, e):
        self.dev_extrinsic = e
        self.log("read LiDAR mount pose", GREEN)

    def ui_view(self):
        s = self.s
        fb_w, _ = self.window.get_size()
        imgui.set_next_window_position(fb_w - 350, 10, imgui.FIRST_USE_EVER)
        imgui.set_next_window_size(340, 760, imgui.FIRST_USE_EVER)
        imgui.begin("View")

        if imgui.radio_button("Live window", not s.map_mode):
            s.map_mode = False
        imgui.same_line()
        if imgui.radio_button("Integration map", s.map_mode):
            s.map_mode = True
        _, s.persist = imgui.slider_float("persistence s", s.persist, 0.05, 120.0, "%.2f", LOG_SLIDER)
        _, s.persist_forever = imgui.checkbox("keep forever", s.persist_forever)
        if s.map_mode:
            # 1 mm floor: voxel keys hold 21 bits per axis, so 1 mm still spans +-1 km.
            changed, mm = imgui.slider_float("voxel mm", s.voxel * 1000.0, 1.0, 500.0, "%.1f", LOG_SLIDER)
            if changed:
                s.voxel = max(mm, 1.0) / 1000.0
            _, s.map_min_count = imgui.slider_int("min hits", s.map_min_count, 1, 50)
            occ = self.pipe.map_occupied / self.pipe.map_cap
            imgui.text_colored(f"map {self.pipe.map_occupied:,} voxels ({occ:.0%})"
                               + (f", {self.pipe.map_dropped:,} dropped" if self.pipe.map_dropped else ""),
                               *(AMBER if occ > 0.7 else GREY))
            if imgui.button("Clear map"):
                self.pipe.clear_map()
        else:
            imgui.text_colored(f"ring {self.pipe.filled:,} / {self.pipe.cap:,} points", *GREY)

        if imgui.collapsing_header("Color", flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]:
            self._ui_color()
        if imgui.collapsing_header("Points", flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]:
            self._ui_points()
        if imgui.collapsing_header("Denoise and filters")[0]:
            self._ui_filters()
        if imgui.collapsing_header("Mount pose (viewer)")[0]:
            self._ui_mount()
        if imgui.collapsing_header("Scene")[0]:
            self._ui_scene()

        imgui.separator()
        if imgui.button("Fit"):
            self.fit_view()
        for label, preset in (("Reset", None), ("Top", "top"), ("Front", "front"), ("Side", "side"),
                              ("Sensor", "sensor")):
            imgui.same_line()
            if imgui.button(label):
                self.camera_preset(preset)
        _, s.freeze = imgui.checkbox("Freeze", s.freeze)
        imgui.same_line()
        if imgui.button("Clear"):
            self.clear_all()
        imgui.same_line()
        if imgui.button("Export PLY"):
            self.act("export PLY", self.export_ply)
        imgui.same_line()
        if imgui.button("Export LAS"):
            self.act("export LAS", self.export_las)
        imgui.same_line()
        if imgui.button("Screenshot"):
            self.screenshot_req = time.strftime("livox_%Y%m%d_%H%M%S.png")
        imgui.end()

    def _ui_color(self):
        s = self.s
        _, s.color_mode = imgui.combo("mode", s.color_mode, gpu.COLOR_MODES)
        if s.color_mode in gpu.SCALAR_MODES:
            _, s.auto_range = imgui.checkbox("auto range", s.auto_range)
            c1, s.color_lo = imgui.drag_float("low", s.color_lo, 0.05)
            c2, s.color_hi = imgui.drag_float("high", s.color_hi, 0.05)
            if c1 or c2:
                s.auto_range = False
        if s.color_mode == gpu.MODE_SOLID:
            _, s.solid = imgui.color_edit3("color", *s.solid)
        if s.color_mode in gpu.NEEDS_NORMALS or s.denoise or s.surfels:
            _, s.radius = imgui.slider_float("neighborhood m", s.radius, 0.02, 1.0, "%.3f", LOG_SLIDER)
        hints = {
            gpu.MODE_GROUND: (not s.ground, "turn on Ground in the Perception panel (G)"),
            gpu.MODE_CLUSTERS: (not s.clusters, "turn on Clusters in the Perception panel (K)"),
            gpu.MODE_SPEED: (not s.clusters, "turn on Clusters (K): colors each tracked cluster by speed"),
            gpu.MODE_MOTION: (True, "red = in a voxel that is new or see-through: moving or just appeared"),
            gpu.MODE_CHANGES: (self.prior is None or not self.prior.localised,
                               "load a prior map and localise (Perception panel): green = matches the scan, "
                               "amber = within 10 cm, red = new or moved"),
        }
        if s.color_mode in hints and hints[s.color_mode][0]:
            imgui.text_colored(hints[s.color_mode][1], *(AMBER if s.color_mode != gpu.MODE_MOTION else GREY))

    def _ui_points(self):
        s = self.s
        _, s.surfels = imgui.checkbox("surfels (discs on PCA normals)", s.surfels)
        if s.surfels:
            _, s.surfel_radius = imgui.slider_float("surfel radius m", s.surfel_radius, 0.005, 0.3, "%.3f",
                                                    LOG_SLIDER)
        else:
            _, s.size_in_metres = imgui.checkbox("size in metres", s.size_in_metres)
            if s.size_in_metres:
                _, s.world_size = imgui.slider_float("size m", s.world_size, 0.002, 0.2, "%.3f", LOG_SLIDER)
            else:
                _, s.point_size = imgui.slider_float("size px", s.point_size, 1.0, 12.0, "%.1f")
            _, s.round_points = imgui.checkbox("round", s.round_points)
            imgui.same_line()
        _, s.edl = imgui.checkbox("eye-dome lighting", s.edl)
        if s.edl:
            _, s.edl_strength = imgui.slider_float("EDL strength", s.edl_strength, 0.0, 3.0)
            _, s.edl_radius = imgui.slider_float("EDL radius px", s.edl_radius, 0.5, 4.0)

    def _ui_filters(self):
        s = self.s
        _, s.denoise = imgui.checkbox("radius outlier removal", s.denoise)
        if s.denoise:
            _, s.min_nbrs = imgui.slider_int("min neighbors", s.min_nbrs, 1, 40)
            _, s.radius = imgui.slider_float("radius m", s.radius, 0.02, 1.0, "%.3f", LOG_SLIDER)
        _, s.min_range = imgui.slider_float("min range m", s.min_range, 0.0, 20.0, "%.2f")
        _, s.max_range = imgui.slider_float("max range m", s.max_range, 1.0, 500.0, "%.1f", LOG_SLIDER)
        _, s.min_refl = imgui.slider_int("min reflectivity", s.min_refl, 0, 255)
        imgui.text("keep noise tags (extended data types):")
        for k, name in enumerate(NOISE_LEVELS):
            _, s.noise_keep[k] = imgui.checkbox(name, s.noise_keep[k])
            if k % 2 == 0:
                imgui.same_line()
        imgui.text("keep returns:")
        for k in range(3):
            imgui.same_line()
            _, s.ret_keep[k] = imgui.checkbox(f"#{k + 1}", s.ret_keep[k])
        _, s.crop_on = imgui.checkbox("crop box (world)", s.crop_on)
        if s.crop_on:
            _, lo = imgui.drag_float3("min", *s.crop_lo, 0.05)
            _, hi = imgui.drag_float3("max", *s.crop_hi, 0.05)
            s.crop_lo, s.crop_hi = list(lo), list(hi)

    def _ui_mount(self):
        s = self.s
        # assign only on change: the widgets return float32-rounded values even when untouched, and a
        # rewritten pose (1.2 -> 1.2000000476837158) would count as a change and clear the map
        c1, rpy = imgui.drag_float3("roll pitch yaw", *s.mount[:3], 0.1)
        c2, xyz = imgui.drag_float3("x y z m", *s.mount[3:], 0.01)
        if c1 or c2:
            s.mount = [*rpy, *xyz]
        if imgui.button("Zero pose"):
            s.mount = [0.0] * 6
        imgui.text_colored("changing the pose clears the map (and restarts odometry)", *GREY)
        imgui.text_colored("remembered for the next start", *GREY)

    def _ui_scene(self):
        s = self.s
        _, s.grid = imgui.checkbox("grid", s.grid)
        imgui.same_line()
        _, s.gizmo = imgui.checkbox("sensor axes / FOV", s.gizmo)
        _, s.grid_z = imgui.drag_float("grid z", s.grid_z, 0.01)
        if imgui.button("Snap grid to floor"):
            z = self.pipe.floor_z()
            if z is not None:
                s.grid_z = round(z, 3)
        _, s.background = imgui.color_edit3("background", *s.background)

    def ui_perception(self):
        fb_w, _ = self.window.get_size()
        imgui.set_next_window_position(fb_w - 700, 10, imgui.FIRST_USE_EVER)
        imgui.set_next_window_size(340, 620, imgui.FIRST_USE_EVER)
        imgui.begin("Perception")
        if imgui.collapsing_header("Odometry (LiDAR-only, scan-to-map ICP)", flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]:
            self._ui_odometry()
        if imgui.collapsing_header("Ground", flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]:
            self._ui_ground()
        if imgui.collapsing_header("Clusters and tracking", flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]:
            self._ui_clusters()
        if imgui.collapsing_header("Motion from map history")[0]:
            self._ui_motion()
        if imgui.collapsing_header("Prior map (a scan of the building)", flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]:
            self.ui_prior()
        imgui.end()

    def _ui_odometry(self):
        s = self.s
        changed, on = imgui.checkbox("odometry (O)", s.odom)
        if changed:
            self.set_odom(on)
        if not s.odom:
            imgui.text_wrapped("Registers 10 Hz frames against a voxel map so a moving Mid-40 builds one "
                               "consistent map without an IMU. Static sensors can leave it off.")
            return
        changed, v = imgui.slider_float("frame s", s.frame_dt, 0.05, 0.5, "%.2f")
        if changed:
            s.frame_dt = v
        changed, mm = imgui.slider_float("map voxel mm", s.map_voxel * 1000.0, 50.0, 600.0, "%.0f", LOG_SLIDER)
        if changed:
            s.map_voxel = mm / 1000.0
        _, s.odom_iters = imgui.slider_int("iterations", s.odom_iters, 3, 30)
        _, s.follow = imgui.checkbox("follow sensor (V)", s.follow)
        imgui.same_line()
        _, s.show_path = imgui.checkbox("path", s.show_path)
        st = self.odom_stats
        p = self.pose_now[:3, 3]
        imgui.text_colored(f"pose  x {p[0]:+.2f}  y {p[1]:+.2f}  z {p[2]:+.2f} m", *GREY)
        col = AMBER if st["weak"] else GREY
        imgui.text_colored(f"frames {st['frames']}  iters {st['iters']}  corr {st['corr']}/{st['ds']}  "
                           f"rms {st['rms'] * 100:.1f} cm", *col)
        held = f"  {st['degen']} direction(s) held by motion prediction" if st.get("degen") else ""
        imgui.text_colored(f"conditioning {st['cond']:.1e}{held}", *col)
        imgui.text_colored(f"{st['ms']:.1f} ms/frame   {st['speed']:.2f} m/s  {st['turn']:.0f} deg/s   "
                           f"map {st['map_voxels']:,} voxels ({st['map_full']:.0%})",
                           *(AMBER if st["map_full"] > 0.7 else GREY))
        if st["skipped"]:
            imgui.text_colored(f"{st['skipped']} frames too sparse to register", *AMBER)
        if st.get("warm"):
            imgui.text_colored("building the first map: hold the sensor still", *AMBER)
        if self.odom_worker is not None:
            ws = self.odom_worker.status()
            dropped = sum(ws["drops"].values())
            imgui.text_colored(f"worker thread: {ws['busy_ms']:.1f} ms/batch, {ws['pending']} queued, "
                               f"{ws['pool_free']} free slots" + (f", {dropped} dropped" if dropped else ""),
                               *(AMBER if dropped or ws["last_error"] else GREY))
            if ws["last_error"]:
                imgui.text_colored(f"worker error: {ws['last_error']}", *RED)
        if imgui.button("Restart odometry"):
            self.clear_all()

    def _ui_ground(self):
        s = self.s
        _, s.ground = imgui.checkbox("ground segmentation (G)", s.ground)
        if s.ground:
            _, s.gnd_cell = imgui.slider_float("cell m", s.gnd_cell, 0.2, 2.0, "%.2f")
            _, s.gnd_thresh = imgui.slider_float("max height m", s.gnd_thresh, 0.03, 0.6, "%.2f")
            _, s.gnd_slope = imgui.slider_float("slope m/cell", s.gnd_slope, 0.0, 1.0, "%.2f")
            _, s.gnd_use_normals = imgui.checkbox("reject steep normals", s.gnd_use_normals)
            imgui.same_line()
            _, s.hide_ground = imgui.checkbox("hide ground", s.hide_ground)
            imgui.text_colored(f"{self.perc.ms['ground']:.2f} ms", *GREY)

    def _ui_clusters(self):
        s = self.s
        _, s.clusters = imgui.checkbox("clusters (K)", s.clusters)
        if s.clusters:
            _, s.cl_voxel = imgui.slider_float("cluster voxel m", s.cl_voxel, 0.03, 0.5, "%.2f", LOG_SLIDER)
            _, s.cl_connect = imgui.slider_float("connect m", s.cl_connect, 0.05, 2.0, "%.2f", LOG_SLIDER)
            _, s.cl_min_pts = imgui.slider_int("min points", s.cl_min_pts, 5, 500)
            _, s.cl_use_ground = imgui.checkbox("exclude ground", s.cl_use_ground)
            if s.cl_use_ground and not s.ground:
                imgui.same_line()
                imgui.text_colored("(turn Ground on)", *AMBER)
            _, s.boxes = imgui.checkbox("boxes + velocity", s.boxes)
            imgui.same_line()
            _, s.labels = imgui.checkbox("labels", s.labels)
            imgui.text_colored(f"{len(self.perc.clusters)} clusters, {len(self.tracks)} confirmed tracks, "
                               f"{self.perc.cc_iters} label passes, {self.perc.ms['clusters']:.2f} ms", *GREY)
            if not self.perc.cc_converged:
                imgui.text_colored("labels did not converge: raise cluster voxel or lower connect", *AMBER)

    def _ui_motion(self):
        s = self.s
        imgui.text_wrapped("The integration map remembers when each voxel first appeared and how often rays "
                           "passed through it. Points in young or see-through voxels are 'moving'.")
        _, s.dyn_settle = imgui.slider_float("settle s", s.dyn_settle, 0.2, 10.0, "%.1f", LOG_SLIDER)
        _, s.hide_static = imgui.checkbox("hide static", s.hide_static)
        imgui.same_line()
        _, s.hide_dynamic = imgui.checkbox("hide moving", s.hide_dynamic)
        _, s.carve = imgui.checkbox("free-space carving (X)", s.carve)
        if s.carve:
            _, s.carve_hide = imgui.checkbox("hide ghost voxels in map", s.carve_hide)
            _, s.carve_ratio = imgui.slider_float("miss/hit ratio", s.carve_ratio, 0.2, 10.0, "%.1f", LOG_SLIDER)
            _, s.carve_min = imgui.slider_int("min misses", s.carve_min, 1, 200)
            _, s.carve_stale = imgui.slider_float("not hit for s", s.carve_stale, 0.1, 10.0, "%.1f", LOG_SLIDER)
            _, s.carve_every = imgui.slider_int("trace every Nth return", s.carve_every, 1, 8)
            imgui.text_colored(f"{self.pipe.ms['carve']:.2f} ms", *GREY)

    def load_prior(self, path: str):
        """Open a prior map (replacing a loaded one); it loads and localises on its own thread."""
        if not path:
            raise ValueError("no prior map given")
        if not os.path.exists(path):
            self.log(f"prior map {path}: not found", RED)
            return
        self.unload_prior()
        self.prior = PriorSession(path, self.device, log=self._prior_log)
        self.prior_path = path
        self.log(f"loading prior map {path}", GREY)

    def unload_prior(self):
        if self.prior is None:
            return
        self.prior.close()
        self.prior = None
        self.gl.set_prior(None)
        self._prior_drawn = False

    def ui_prior(self):
        s = self.s
        pr = self.prior
        if pr is None:
            imgui.text_wrapped("Localise the scanner in an earlier scan of the place, see what changed, and hold "
                               "the odometry to it.")
            _, self.prior_path = imgui.input_text("npz", self.prior_path, 512)
            if not self.prior_path:
                imgui.text_colored(PRIOR_HINT, *GREY)
            if imgui.button("Load prior map"):
                if self.prior_path:
                    self.act("load prior map", lambda: self.load_prior(self.prior_path))
                else:
                    self.log(f"enter the path of a prior map first ({PRIOR_HINT})", AMBER)
            return
        st = pr.status()
        color = {"tracking": GREEN, "lost": AMBER, "ambiguous": AMBER, "error": RED}.get(st["state"], GREY)
        imgui.text_colored(f"{os.path.basename(pr.path)}: {st['state']}", *color)
        if st["detail"]:
            imgui.text_wrapped(st["detail"])
        if st["points"]:
            imgui.text_colored(f"{st['points']:,} points, {st['mib']:.0f} MiB grid; loaded in {st['load_s']:.1f} s, "
                               f"built in {st['build_s']:.2f} s", *GREY)
        if pr.localised:
            T = pr.T_MW @ self.pose_now
            yaw = math.degrees(math.atan2(T[1, 0], T[0, 0]))
            imgui.text(f"sensor in the map: {T[0, 3]:.2f} {T[1, 3]:.2f} {T[2, 3]:.2f} m, heading {yaw:.0f} deg")
            imgui.text_colored(f"fit {st['fit']:.2f} (points within 3 cm), {st['corrections']} corrections, "
                               f"{st['rejected']} rejected", *GREY)
        if pr.ready:
            if imgui.button("Localise now"):
                pr.localise_now()
            imgui.same_line()
            if imgui.button("Forget pose"):
                pr.forget()
            if pr.state == "ambiguous" or (pr.state == "lost" and pr.has_pending_pose):
                imgui.same_line()
                if imgui.button("Use it anyway"):
                    pr.accept_ambiguous()
            if not s.odom:
                imgui.text_colored("without odometry: hold the scanner still for 2 s to localise", *GREY)
            _, pr.auto = imgui.checkbox("localise automatically", pr.auto)
            imgui.same_line()
            _, pr.track_on = imgui.checkbox("track drift", pr.track_on)
            _, s.prior_draw = imgui.checkbox("draw the scan", s.prior_draw)
            if s.prior_draw:
                imgui.same_line()
                imgui.push_item_width(80)
                _, s.prior_point_size = imgui.slider_float("px", s.prior_point_size, 1.0, 4.0, "%.1f")
                imgui.pop_item_width()
            if s.color_mode != gpu.MODE_CHANGES:
                if imgui.button("Show changes (Changes colours)"):
                    s.color_mode = gpu.MODE_CHANGES
            else:
                c1, near = imgui.slider_float("matches below cm", s.chg_near * 100, 1.0, 10.0, "%.1f")
                c2, far = imgui.slider_float("new above cm", s.chg_far * 100, 3.0, 50.0, "%.0f")
                if c1:
                    s.chg_near = near / 100
                if c2:
                    s.chg_far = max(far / 100, s.chg_near + 0.005)
        if imgui.button("Unload"):
            self.act("unload prior map", self.unload_prior)

    def ui_stats(self):
        _, fb_h = self.window.get_size()
        imgui.set_next_window_position(10, fb_h - 210, imgui.FIRST_USE_EVER)
        imgui.set_next_window_size(390, 200, imgui.FIRST_USE_EVER)
        imgui.begin("Stats")
        src = self.source.label if self.source else "no source"
        imgui.text(src)
        imgui.text(f"{self.fps:5.0f} fps   frame {self.frame_ms:5.1f} ms   Warp GPU {self.gpu_ms:5.2f} ms")
        imgui.text(f"visible {self.pipe.count:,} points   {self.pps:,.0f} pts/s")
        st = self.stats
        if st:
            lost, bad = st.get("lost_packets", 0), st.get("bad_packets", 0)
            imgui.text_colored(
                f"packets {st.get('packets', 0):,}  lost {lost}  bad {bad}  data type {st.get('data_type', '?')}",
                *(AMBER if lost or bad else GREY),
            )
            if st.get("dropped_points"):
                imgui.text_colored(f"dropped in host buffer: {st['dropped_points']:,}", *AMBER)
            if st.get("problems"):
                imgui.text_colored("status: " + ", ".join(st["problems"]), *AMBER)
        if self.s.odom:
            o = self.odom_stats
            imgui.text_colored(f"odometry {o['frames']} frames  {o['ms']:.1f} ms  rms {o['rms'] * 100:.1f} cm"
                               + ("  WEAK geometry" if o["weak"] else ""), *(AMBER if o["weak"] else GREY))
        if self.gl.interop:
            imgui.text_colored("CUDA-GL interop: zero-copy to GL", *GREEN)
        else:
            imgui.text_colored("CPU FALLBACK: no CUDA-GL interop", *RED)
        imgui.plot_lines("##rate", self.rate_hist, scale_min=0.0, graph_size=(370, 36))
        imgui.end()

    def _export_common(self):
        n = self.pipe.count
        if n == 0:
            self.log("nothing to export", AMBER)
            return None
        xyz, attr, t = self.pipe.export()
        rgba = self.gl.read_colors(n)
        keep = rgba[:, 3] > 0
        gnd = None
        if self.s.ground:
            gnd, _ = self.pipe.export_labels()
            gnd = gnd[keep]
        return xyz[keep], rgba[keep], (attr[keep] & 0xFF).astype(np.uint8), gnd, t[keep]

    def export_ply(self):
        r = self._export_common()
        if r is None:
            return
        xyz, rgba, refl, _, _ = r
        path = time.strftime("livox_%Y%m%d_%H%M%S.ply")
        export.write_ply(path, xyz, rgba, refl)
        self.log(f"exported {len(xyz):,} points to {path}", GREEN)

    def export_las(self):
        r = self._export_common()
        if r is None:
            return
        xyz, rgba, refl, gnd, t = r
        path = time.strftime("livox_%Y%m%d_%H%M%S.las")
        try:
            export.write_las(path, xyz, rgba, refl, gnd, t)
        except ImportError:
            self.log("LAS export needs laspy (pip install laspy)", RED)
            return
        self.log(f"exported {len(xyz):,} points to {path}" + (" with ground classification" if gnd is not None else ""),
                 GREEN)

    # ---- input ---------------------------------------------------------------------------

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if imgui.get_io().want_capture_mouse:
            return
        self.sensor_view = False  # moving the camera by hand leaves the Sensor view
        if buttons & pyglet.window.mouse.LEFT and not modifiers & pyglet.window.key.MOD_SHIFT:
            self.cam.orbit(dx, dy)
        else:  # right or middle button, or shift + left
            self.cam.pan(dx, dy, self.window.height)

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        if imgui.get_io().want_capture_mouse:
            return
        self.sensor_view = False
        self.cam.zoom(scroll_y)

    def on_key_press(self, symbol, modifiers):
        if imgui.get_io().want_capture_keyboard:
            return
        k = pyglet.window.key
        s = self.s
        if symbol == k.F:
            s.freeze = not s.freeze
        elif symbol == k.C:
            self.clear_all()
        elif symbol == k.M:
            s.map_mode = not s.map_mode
        elif symbol == k.E:
            s.edl = not s.edl
        elif symbol == k.D:
            s.denoise = not s.denoise
        elif symbol == k.O:
            self.set_odom(not s.odom)
        elif symbol == k.G:
            s.ground = not s.ground
        elif symbol == k.K:
            s.clusters = not s.clusters
        elif symbol == k.S:
            s.surfels = not s.surfels
        elif symbol == k.X:
            s.carve = not s.carve
        elif symbol == k.V:
            s.follow = not s.follow
        elif symbol == k.R:
            self.camera_preset(None)
        elif symbol == k.T:
            self.camera_preset("top")
        elif symbol == k.P:
            self.screenshot_req = time.strftime("livox_%Y%m%d_%H%M%S.png")
        elif k._1 <= symbol <= k._9:
            s.color_mode = min(symbol - k._1, gpu.MODE_SOLID)
        elif symbol == k._0:
            s.color_mode = gpu.MODE_SOLID

    # ---- loop ----------------------------------------------------------------------------

    def run(self):
        last = time.perf_counter()
        fps_t, fps_n = last, 0
        while not self.window.has_exit:
            pyglet.clock.tick()
            self.window.switch_to()
            self.window.dispatch_events()
            if self.window.has_exit:
                break
            last_frame = bool(self.args.frames) and self.frames + 1 >= self.args.frames
            if last_frame and self.args.screenshot:
                self.screenshot_req = self.args.screenshot
            t0 = time.perf_counter()
            try:
                self.frame()
            except Exception:  # noqa: BLE001 - keep the viewer alive, show the error
                traceback.print_exc()
                self.log("frame error (see console)", RED)
                self._recover_imgui()
            self.frame_ms = (time.perf_counter() - t0) * 1e3
            self.window.flip()
            self.frames += 1
            fps_n += 1
            now = time.perf_counter()
            if now - fps_t >= 0.5:
                self.fps = fps_n / (now - fps_t)
                fps_t, fps_n = now, 0
            if last_frame:
                break
        self.shutdown()

    def _save_mount(self):
        self.config["mount"] = [float(v) for v in self.s.mount]
        try:
            save_config(self.config_path, self.config)
            self.log(f"mount pose saved to {self.config_path}", GREY)
        except OSError as e:
            self.log(f"could not save the mount pose: {e}", AMBER)
        self.mount_saved = tuple(self.s.mount)
        self.mount_dirty_at = None

    def _recover_imgui(self):
        """Close the broken frame. An error inside a window leaves it open and end_frame() then
        asserts; in that case rebuild the imgui context so the next frame starts clean."""
        try:
            imgui.end_frame()
            return
        except Exception:  # noqa: BLE001
            pass
        try:
            self.imgui.shutdown()
        except Exception:  # noqa: BLE001
            pass
        imgui.destroy_context()
        imgui.create_context()
        self.imgui = create_renderer(self.window)
        imgui.get_io().font_global_scale = self.args.ui_scale
        imgui.style_colors_dark()

    def camera_preset(self, preset):
        # "sensor" stays on (every frame, from the current pose) until the camera is moved or another view is picked
        self.sensor_view = preset == "sensor"
        if preset is None:
            self.cam.reset()
        elif preset == "sensor":
            T = self.pose_now
            self.cam.look_from(T[:3, 3], T[:3, 0], 3.0)
        else:
            self.cam.preset(preset)

    def shutdown(self):
        if tuple(self.s.mount) != self.mount_saved:
            self._save_mount()
        if self.source is not None:
            try:
                self.source.close()
            except Exception:  # noqa: BLE001
                pass
        if self.odom_worker is not None:
            try:
                self.odom_worker.close()
            except Exception:  # noqa: BLE001
                pass
        if self.prior is not None:
            try:
                self.prior.close()
            except Exception:  # noqa: BLE001
                pass
        try:
            self.window.switch_to()
            self.imgui.shutdown()
        except Exception:  # noqa: BLE001 - GL teardown order at exit; the process frees it anyway
            pass
        self.window.close()


def _hue(k: int):
    """Same golden-ratio hue walk as the shader, for overlay lines and labels."""
    x = k * 0.6180339887 + 0.11
    h = (x - math.floor(x)) * 6.0
    i = int(h)
    f = h - i
    s = 0.72
    p, q, t = 1.0 - s, 1.0 - s * f, 1.0 - s * (1.0 - f)
    return [(1.0, t, p), (q, 1.0, p), (p, 1.0, t), (p, q, 1.0), (t, p, 1.0), (1.0, p, q)][i % 6]


def parse_size(s: str):
    w, h = s.lower().split("x")
    return int(w), int(h)


def main(argv=None):
    p = argparse.ArgumentParser(prog="livox_warp", description="Livox LiDAR viewer on NVIDIA Warp")
    src = p.add_argument_group("source (default: auto-connect to the first LiDAR discovered)")
    src.add_argument("--lidar", help="LiDAR IP to connect to")
    src.add_argument("--host", help="local IP the LiDAR streams to (default: the one on its subnet)")
    src.add_argument("--type", type=int, default=DEV_UNKNOWN,
                     help="device type (1 Mid-40, 2 Tele-15, 3 Horizon, 6 Mid-70, 7 Avia; default: from discovery)")
    src.add_argument("--replay", help="play an LVXR recording (the viewer's own raw-packet format)")
    src.add_argument("--speed", type=float, default=1.0, help="replay speed factor")
    src.add_argument("--sim", action="store_true", help="Warp-simulated Mid-40, no hardware")
    src.add_argument("--sim-motion", choices=SIM_MOTIONS, default="static", help="how the simulated sensor moves")
    src.add_argument("--sim-scene", choices=SIM_SCENES, default="box",
                     help="bare room, or the room furnished as an office")
    src.add_argument("--no-auto", action="store_true", help="don't auto-connect to a discovered LiDAR")
    start = p.add_argument_group("initial state (all of it can be changed in the panels)")
    start.add_argument("--odom", action="store_true", help="start with LiDAR-only odometry on")
    start.add_argument("--ground", action="store_true", help="start with ground segmentation on")
    start.add_argument("--clusters", action="store_true", help="start with clustering and tracking on")
    start.add_argument("--carve", action="store_true", help="start with free-space carving on")
    start.add_argument("--surfels", action="store_true", help="start with surfel rendering on")
    start.add_argument("--map-mode", action="store_true", help="start in integration-map mode")
    start.add_argument("--color", type=str.lower, choices=COLOR_NAMES, metavar="MODE",
                       help="initial color mode: " + ", ".join(COLOR_NAMES))
    start.add_argument("--persist", type=float, help="initial persistence of the live window in seconds")
    start.add_argument("--view", choices=["top", "front", "side", "sensor"], help="camera preset after auto-fit")
    start.add_argument("--prior-map", metavar="NPZ",
                       help="prior map to localise in (from python -m livox_warp.prior_map convert scan.e57); "
                            "enables the Changes colours")
    sys_ = p.add_argument_group("system")
    sys_.add_argument("--device", help="Warp device for the pipeline, perception, odometry and prior map "
                                       "(default: cuda:0 if CUDA is available, else cpu)")
    sys_.add_argument("--ring", type=int, default=1 << 23, help="live ring capacity in points")
    sys_.add_argument("--map-slots", type=int, default=1 << 23, help="integration map slots (power of two)")
    sys_.add_argument("--size", type=parse_size, default=(1600, 950), help="window size WxH")
    sys_.add_argument("--ui-scale", type=float, default=1.15, help="imgui font scale (1.0 = native size)")
    sys_.add_argument("--no-vsync", action="store_true", help="don't wait for the display's vertical sync")
    sys_.add_argument("--config", metavar="JSON",
                      help=f"settings file (default {CONFIG_PATH}); only the viewer mount pose is stored in it")
    cap = p.add_argument_group("capture (tests and README animations)")
    cap.add_argument("--frames", type=int, default=0, help="exit after N frames")
    cap.add_argument("--screenshot", metavar="PNG", help="save the last frame (with --frames)")
    cap.add_argument("--frames-dir", metavar="DIR",
                     help="save every rendered frame as DIR/frame_000001.png, frame_000002.png, ...; "
                          "with --frames N this yields a sequence to assemble into a GIF offline")
    cap.add_argument("--frames-every", type=int, default=1, metavar="N",
                     help="with --frames-dir: save every Nth frame only (default 1)")
    args = p.parse_args(argv)
    if args.frames_every < 1:
        p.error("--frames-every must be at least 1")
    App(args).run()


if __name__ == "__main__":
    main()
