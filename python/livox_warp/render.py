"""OpenGL side: vertex buffers Warp writes into via CUDA-GL interop, points or surfels, eye-dome lighting,
the ground grid, the sensor gizmo (axes and FOV cone), a prior map drawn in its own frame, overlay lines
(trajectory, boxes, velocity arrows), and the orbit camera with its presets and projection."""

from __future__ import annotations

import math
from dataclasses import dataclass

import moderngl
import numpy as np
import warp as wp

MAX_PITCH = 1.55  # rad, just short of straight up or down, where the look-at basis degenerates
ORBIT_RADIANS_PER_PIXEL = 0.006
ZOOM_STEP = 0.88  # camera distance factor per wheel notch
MIN_DIST, MAX_DIST = 0.05, 2000.0

POINT_VS = """
#version 330
uniform mat4 mvp;
uniform float point_size;
uniform float world_size;   // > 0: size points in metres instead of pixels
uniform float viewport_h;
uniform float proj_scale;   // 1 / tan(fov / 2)
in vec3 in_pos;
in vec4 in_color;
out vec3 v_color;
void main() {
    if (in_color.a == 0.0) {                 // filtered by a Warp kernel: cull
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        gl_PointSize = 0.0;
        v_color = vec3(0.0);
        return;
    }
    gl_Position = mvp * vec4(in_pos, 1.0);
    float s = point_size;
    if (world_size > 0.0) {
        s = world_size * proj_scale * viewport_h * 0.5 / max(gl_Position.w, 1e-3);
    }
    gl_PointSize = clamp(s, 1.0, 64.0);
    v_color = in_color.rgb;
}
"""

POINT_FS = """
#version 330
uniform int round_points;
in vec3 v_color;
out vec4 frag;
void main() {
    if (round_points == 1) {
        vec2 d = gl_PointCoord * 2.0 - 1.0;
        if (dot(d, d) > 1.0) discard;
    }
    frag = vec4(v_color, 1.0);
}
"""

# Surfels: each point becomes a disc in its tangent plane (PCA normal), so a sparse scan reads as a
# closed surface. The geometry shader builds the quad; the fragment shader cuts the disc.
SURFEL_VS = """
#version 330
in vec3 in_pos;
in vec3 in_nrm;
in vec4 in_color;
out vec3 v_nrm;
out vec3 v_col;
out float v_a;
void main() {
    gl_Position = vec4(in_pos, 1.0);
    v_nrm = in_nrm;
    v_col = in_color.rgb;
    v_a = in_color.a;
}
"""

SURFEL_GS = """
#version 330
layout(points) in;
layout(triangle_strip, max_vertices = 4) out;
uniform mat4 mvp;
uniform float radius;
uniform vec3 eye;
in vec3 v_nrm[];
in vec3 v_col[];
in float v_a[];
out vec3 g_col;
out vec2 g_uv;
void main() {
    if (v_a[0] == 0.0) return;
    vec3 p = gl_in[0].gl_Position.xyz;
    vec3 n = v_nrm[0];
    float l = length(n);
    n = l > 1e-6 ? n / l : normalize(eye - p);
    vec3 a = abs(n.x) < 0.9 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);
    vec3 t1 = normalize(cross(n, a));
    vec3 t2 = cross(n, t1);
    vec3 col = v_col[0] * (0.7 + 0.3 * abs(dot(n, normalize(eye - p))));
    vec2 uv[4] = vec2[4](vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0), vec2(1.0, 1.0));
    for (int k = 0; k < 4; k++) {
        vec3 q = p + radius * (uv[k].x * t1 + uv[k].y * t2);
        gl_Position = mvp * vec4(q, 1.0);
        g_col = col;
        g_uv = uv[k];
        EmitVertex();
    }
    EndPrimitive();
}
"""

SURFEL_FS = """
#version 330
in vec3 g_col;
in vec2 g_uv;
out vec4 frag;
void main() {
    if (dot(g_uv, g_uv) > 1.0) discard;
    frag = vec4(g_col, 1.0);
}
"""

LINE_VS = """
#version 330
uniform mat4 mvp;
in vec3 in_pos;
in vec3 in_color;
out vec3 v_color;
void main() { gl_Position = mvp * vec4(in_pos, 1.0); v_color = in_color; }
"""

LINE_FS = """
#version 330
in vec3 v_color;
out vec4 frag;
void main() { frag = vec4(v_color, 1.0); }
"""

# Eye-dome lighting: shade each pixel by how much its neighbours sit in front of it (log depth).
EDL_VS = """
#version 330
out vec2 uv;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

EDL_FS = """
#version 330
uniform sampler2D color_tex;
uniform sampler2D depth_tex;
uniform vec2 texel;
uniform float strength;
uniform float radius;
uniform float near;
uniform float far;
uniform int enabled;
in vec2 uv;
out vec4 frag;
float lin(float d) { float z = d * 2.0 - 1.0; return 2.0 * near * far / (far + near - z * (far - near)); }
void main() {
    vec4 c = texture(color_tex, uv);
    float d = texture(depth_tex, uv).r;
    if (enabled == 0 || d >= 1.0) { frag = vec4(c.rgb, 1.0); return; }
    float ld = log2(lin(d));
    float sum = 0.0;
    for (int k = 0; k < 8; k++) {
        float a = 6.2831853 * float(k) / 8.0;
        vec2 off = vec2(cos(a), sin(a)) * radius * texel;
        float dn = texture(depth_tex, uv + off).r;
        float ln = dn >= 1.0 ? log2(far) : log2(lin(dn));
        sum += max(0.0, ld - ln);
    }
    float shade = exp(-sum * strength * 30.0);  // 30: strength 0..2 spans subtle to strong
    frag = vec4(c.rgb * shade, 1.0);
}
"""


def look_at(eye, target, up):
    f = target - eye
    f = f / max(np.linalg.norm(f), 1e-9)
    s = np.cross(f, up)
    s = s / max(np.linalg.norm(s), 1e-9)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float64)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def perspective(fov_deg, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fov_deg) / 2)
    m = np.zeros((4, 4))
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2 * far * near / (near - far)
    m[3, 2] = -1
    return m


class OrbitCamera:
    """Z-up orbit camera. Livox frame: x forward, y left, z up."""

    def __init__(self):
        self.fov = 55.0
        self.near, self.far = 0.05, 3000.0
        self.reset()

    def reset(self, target=(4.0, 0.0, 0.0), dist=10.0):
        self.target = np.array(target, dtype=np.float64)
        self.yaw = math.pi  # eye behind the sensor, looking down +x
        self.pitch = math.radians(22)
        self.dist = dist

    def preset(self, name: str):
        if name == "top":
            self.yaw, self.pitch = math.pi, math.radians(89.0)
        elif name == "front":
            self.yaw, self.pitch = math.pi, math.radians(4.0)
        elif name == "side":
            self.yaw, self.pitch = math.pi / 2, math.radians(4.0)
        elif name == "sensor":
            self.target = np.array([self.dist, 0.0, 0.0])
            self.yaw, self.pitch = math.pi, 0.0

    @property
    def eye(self):
        cp = math.cos(self.pitch)
        return self.target + self.dist * np.array(
            [cp * math.cos(self.yaw), cp * math.sin(self.yaw), math.sin(self.pitch)]
        )

    def matrices(self, aspect):
        view = look_at(self.eye, self.target, np.array([0.0, 0.0, 1.0]))
        proj = perspective(self.fov, aspect, self.near, self.far)
        return view, proj

    def look_from(self, pos, forward, dist: float = 5.0):
        """Put the eye at `pos` looking along `forward` (orbit target `dist` ahead of it)."""
        f = np.asarray(forward, float)
        f = f / max(np.linalg.norm(f), 1e-9)
        self.dist = dist
        self.target = np.asarray(pos, float) + f * dist
        self.pitch = float(np.clip(math.asin(-f[2]), -MAX_PITCH, MAX_PITCH))
        self.yaw = math.atan2(-f[1], -f[0])

    def orbit(self, dx, dy):
        self.yaw -= dx * ORBIT_RADIANS_PER_PIXEL
        self.pitch = float(np.clip(self.pitch - dy * ORBIT_RADIANS_PER_PIXEL, -MAX_PITCH, MAX_PITCH))

    def fit(self, lo, hi):
        """Frame an axis-aligned box, keeping the current viewing direction."""
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        self.target = (lo + hi) / 2
        radius = max(np.linalg.norm(hi - lo) / 2, 0.5)
        self.dist = radius / math.sin(math.radians(self.fov) / 2) * 0.9

    def pan(self, dx, dy, viewport_h):
        view = look_at(self.eye, self.target, np.array([0.0, 0.0, 1.0]))
        right, up = view[0, :3], view[1, :3]
        scale = 2 * self.dist * math.tan(math.radians(self.fov) / 2) / max(viewport_h, 1)
        self.target += (-dx * right - dy * up) * scale

    def zoom(self, steps):
        self.dist = float(np.clip(self.dist * (ZOOM_STEP**steps), MIN_DIST, MAX_DIST))

    def project(self, pts: np.ndarray, size):
        """World points -> (x, y) window pixels (origin top-left) and a visibility mask."""
        w, h = size
        view, proj = self.matrices(w / max(h, 1))
        m = proj @ view
        p = np.concatenate([np.asarray(pts, float).reshape(-1, 3), np.ones((len(pts), 1))], axis=1) @ m.T
        ok = p[:, 3] > 1e-3
        ndc = p[:, :2] / np.maximum(p[:, 3:4], 1e-3)
        x = (ndc[:, 0] * 0.5 + 0.5) * w
        y = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * h
        ok &= (ndc[:, 0] > -1.1) & (ndc[:, 0] < 1.1) & (ndc[:, 1] > -1.1) & (ndc[:, 1] < 1.1)
        return x, y, ok


def _gl_mat(m: np.ndarray) -> bytes:
    return m.T.astype("f4").tobytes()  # column-major for GLSL


def box_lines(lo, hi, color) -> list:
    """12 edges of an axis-aligned box as (x y z r g b) line-vertex tuples."""
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    c = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    out = []
    for a, b in edges:
        out.append((*c[a], *color))
        out.append((*c[b], *color))
    return out


def arrow_lines(origin, vec, color, head: float = 0.15) -> list:
    """A line with a small V head, for velocity vectors."""
    o = np.asarray(origin, float)
    v = np.asarray(vec, float)
    L = np.linalg.norm(v)
    if L < 1e-3:
        return []
    tip = o + v
    d = v / L
    side = np.cross(d, (0.0, 0.0, 1.0))
    if np.linalg.norm(side) < 1e-3:
        side = np.array([1.0, 0.0, 0.0])
    side /= np.linalg.norm(side)
    h = min(head, 0.4 * L)
    out = [(*o, *color), (*tip, *color)]
    for s in (1.0, -1.0):
        out += [(*tip, *color), (*(tip - d * h + side * s * h * 0.5), *color)]
    return out


@dataclass
class DrawOptions:
    """What Renderer.draw shows and how."""

    background: tuple[float, float, float] = (0.06, 0.06, 0.08)
    grid: bool = True
    gizmo: bool = True
    point_size: float = 2.0  # pixels, unless size_in_metres
    size_in_metres: bool = False
    world_size: float = 0.02  # metres per point when size_in_metres
    round_points: bool = True
    edl: bool = True  # eye-dome lighting
    edl_strength: float = 0.7
    edl_radius: float = 1.5  # pixels
    surfels: bool = False  # discs on the PCA normals instead of points (needs normals)
    surfel_radius: float = 0.03  # metres
    prior: tuple[np.ndarray, float] | None = None  # (prior-map -> world matrix, point size px) to draw the prior map


class Renderer:
    """Owns the GL buffers and programs; Warp writes the point buffers, draw() composes a frame."""

    def __init__(self, ctx: moderngl.Context, capacity: int, device):
        self.ctx = ctx
        self.capacity = capacity
        self.device = device

        self.vbo_pos = ctx.buffer(reserve=capacity * 12, dynamic=True)
        self.vbo_col = ctx.buffer(reserve=capacity * 4, dynamic=True)
        self.vbo_nrm = ctx.buffer(reserve=capacity * 12, dynamic=True)
        self.point_prog = ctx.program(vertex_shader=POINT_VS, fragment_shader=POINT_FS)
        self.point_vao = ctx.vertex_array(
            self.point_prog, [(self.vbo_pos, "3f", "in_pos"), (self.vbo_col, "4f1", "in_color")]
        )
        self.surfel_prog = ctx.program(vertex_shader=SURFEL_VS, geometry_shader=SURFEL_GS, fragment_shader=SURFEL_FS)
        self.surfel_vao = ctx.vertex_array(
            self.surfel_prog,
            [(self.vbo_pos, "3f", "in_pos"), (self.vbo_nrm, "3f", "in_nrm"), (self.vbo_col, "4f1", "in_color")],
        )
        flags = wp.RegisteredGLBuffer.WRITE_DISCARD
        self.reg_pos = wp.RegisteredGLBuffer(self.vbo_pos.glo, device, flags)
        self.reg_col = wp.RegisteredGLBuffer(self.vbo_col.glo, device, flags)
        self.reg_nrm = wp.RegisteredGLBuffer(self.vbo_nrm.glo, device, flags)
        # True when Warp writes straight into GL memory; False means Warp fell back to host copies.
        self.interop = (self.reg_pos.resource is not None and self.reg_col.resource is not None
                        and self.reg_nrm.resource is not None)

        self.line_prog = ctx.program(vertex_shader=LINE_VS, fragment_shader=LINE_FS)
        self.grid_vbo = None
        self.grid_vao = None
        self.grid_key = None
        self.grid_n = 0
        self.gizmo_vbo = ctx.buffer(reserve=6 * 4 * 128, dynamic=True)  # 128 line vertices; set_gizmo resizes
        self.gizmo_vao = ctx.vertex_array(self.line_prog, [(self.gizmo_vbo, "3f 3f", "in_pos", "in_color")])
        self.gizmo_n = 0
        self.overlay_vbo = ctx.buffer(reserve=6 * 4 * 4096, dynamic=True)
        self.overlay_vao = ctx.vertex_array(self.line_prog, [(self.overlay_vbo, "3f 3f", "in_pos", "in_color")])
        self.overlay_n = 0
        # a prior map (an earlier scan of the place), drawn in its own frame through a model matrix (set_prior)
        self.prior_vbo = None
        self.prior_vao = None
        self.prior_n = 0

        self.edl_prog = ctx.program(vertex_shader=EDL_VS, fragment_shader=EDL_FS)
        self.edl_vao = ctx.vertex_array(self.edl_prog, [])
        self.fbo = None
        self.fbo_size = (0, 0)
        self.color_tex = None
        self.depth_tex = None

    # ---- Warp interop ---------------------------------------------------------------------

    def map(self):
        return (
            self.reg_pos.map(dtype=wp.vec3, shape=(self.capacity,)),
            self.reg_col.map(dtype=wp.uint32, shape=(self.capacity,)),
            self.reg_nrm.map(dtype=wp.vec3, shape=(self.capacity,)),
        )

    def unmap(self):
        self.reg_pos.unmap()
        self.reg_col.unmap()
        self.reg_nrm.unmap()

    def read_colors(self, n: int) -> np.ndarray:
        """RGBA8 of the first n points, for export."""
        raw = np.frombuffer(self.vbo_col.read(size=n * 4), dtype=np.uint8)
        return raw.reshape(-1, 4)

    # ---- scene helpers --------------------------------------------------------------------

    def set_grid(self, z: float, extent: int = 60, step: float = 1.0):
        key = (round(z, 3), extent, step)
        if key == self.grid_key:
            return
        self.grid_key = key
        lines = []
        for k in range(-extent, extent + 1):
            v = k * step
            major = k % 10 == 0
            c = (0.32, 0.34, 0.38) if major else (0.19, 0.20, 0.23)
            if k == 0:
                c = (0.40, 0.42, 0.48)
            lines += [(-extent * step, v, z, *c), (extent * step, v, z, *c)]
            lines += [(v, -extent * step, z, *c), (v, extent * step, z, *c)]
        data = np.array(lines, dtype="f4")
        if self.grid_vbo is not None:
            self.grid_vbo.release()
            self.grid_vao.release()
        self.grid_vbo = self.ctx.buffer(data.tobytes())
        self.grid_vao = self.ctx.vertex_array(self.line_prog, [(self.grid_vbo, "3f 3f", "in_pos", "in_color")])
        self.grid_n = len(data)

    def set_gizmo(self, mount: np.ndarray, fov_deg: float | None, fov_range: float = 5.0):
        """Sensor axes (x red, y green, z blue) and, for circular-FOV LiDARs, the FOV cone."""
        o = mount[:3, 3]
        ax = mount[:3, :3]
        verts = []
        for k, c in enumerate([(1, 0.25, 0.25), (0.3, 1, 0.3), (0.35, 0.55, 1)]):
            verts += [(*o, *c), (*(o + ax[:, k] * 0.6), *c)]
        if fov_deg is not None and fov_deg > 0.0:
            r = fov_range * math.tan(math.radians(fov_deg / 2))
            ring = []
            for i in range(49):
                a = 2 * math.pi * i / 48
                ring.append(o + ax @ np.array([fov_range, r * math.cos(a), r * math.sin(a)]))
            c = (0.55, 0.5, 0.2)
            for i in range(48):
                verts += [(*ring[i], *c), (*ring[i + 1], *c)]
            for i in range(0, 48, 12):
                verts += [(*o, *c), (*ring[i], *c)]
        data = np.array(verts, dtype="f4")
        self.gizmo_vbo.orphan(size=data.nbytes)
        self.gizmo_vbo.write(data.tobytes())
        self.gizmo_n = len(data)

    def set_overlay(self, verts):
        """Line list drawn over the scene: an (N, 6) array of x y z r g b, N even, or None."""
        if verts is None or len(verts) == 0:
            self.overlay_n = 0
            return
        data = np.ascontiguousarray(verts, dtype="f4")
        if data.nbytes > self.overlay_vbo.size:
            self.overlay_vbo.orphan(size=data.nbytes * 2)
        self.overlay_vbo.write(data.tobytes())
        self.overlay_n = len(data)

    # ---- frame ----------------------------------------------------------------------------

    def set_prior(self, xyz: np.ndarray | None, rgba: np.ndarray | None = None):
        """Upload a static point set (the prior map, in its own frame) to draw with a model matrix."""
        if self.prior_vbo is not None:
            self.prior_vao.release()
            self.prior_vbo.release()
            self.prior_vbo = self.prior_vao = None
            self.prior_n = 0
        if xyz is None or len(xyz) == 0:
            return
        data = np.empty(len(xyz), dtype=[("p", np.float32, 3), ("c", np.uint8, 4)])
        data["p"] = xyz
        data["c"] = rgba if rgba is not None else 180
        self.prior_vbo = self.ctx.buffer(data.tobytes())
        self.prior_vao = self.ctx.vertex_array(self.point_prog, [(self.prior_vbo, "3f 4f1", "in_pos", "in_color")])
        self.prior_n = len(xyz)

    def _ensure_fbo(self, size):
        if self.fbo is not None and self.fbo_size == size:
            return
        if self.fbo is not None:
            self.fbo.release()
            self.color_tex.release()
            self.depth_tex.release()
        self.color_tex = self.ctx.texture(size, 4)
        self.depth_tex = self.ctx.depth_texture(size)
        self.fbo = self.ctx.framebuffer([self.color_tex], self.depth_tex)
        self.fbo_size = size

    def _use_points(self, mvp: bytes, cam: OrbitCamera, viewport_h: int, size_px: float, world_size: float,
                    round_points: bool):
        """Set the point program's uniforms; world_size > 0 sizes points in metres instead of pixels."""
        p = self.point_prog
        p["mvp"].write(mvp)
        p["point_size"].value = float(size_px)
        p["world_size"].value = float(world_size)
        p["viewport_h"].value = float(viewport_h)
        p["proj_scale"].value = 1.0 / math.tan(math.radians(cam.fov) / 2)
        p["round_points"].value = 1 if round_points else 0

    def draw(self, count: int, cam: OrbitCamera, size, opts: DrawOptions):
        w, h = size
        if w <= 0 or h <= 0:
            return
        self._ensure_fbo((w, h))
        view, proj = cam.matrices(w / h)
        mvp = _gl_mat(proj @ view)

        self.fbo.use()
        self.ctx.viewport = (0, 0, w, h)
        bg = opts.background
        self.fbo.clear(bg[0], bg[1], bg[2], 1.0, depth=1.0)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.PROGRAM_POINT_SIZE)

        self.line_prog["mvp"].write(mvp)
        if opts.grid and self.grid_vao is not None:
            self.grid_vao.render(moderngl.LINES, vertices=self.grid_n)
        if opts.gizmo and self.gizmo_n:
            self.gizmo_vao.render(moderngl.LINES, vertices=self.gizmo_n)

        if count > 0:
            n = min(count, self.capacity)
            if opts.surfels:
                p = self.surfel_prog
                p["mvp"].write(mvp)
                p["radius"].value = float(opts.surfel_radius)
                p["eye"].value = tuple(float(v) for v in cam.eye)
                self.surfel_vao.render(moderngl.POINTS, vertices=n)
            else:
                self._use_points(mvp, cam, h, opts.point_size, opts.world_size if opts.size_in_metres else 0.0,
                                 opts.round_points)
                self.point_vao.render(moderngl.POINTS, vertices=n)

        if opts.prior is not None and self.prior_n:
            T_WM, size_px = opts.prior  # prior map -> world
            self._use_points(_gl_mat(proj @ view @ np.asarray(T_WM, dtype=np.float64)), cam, h, size_px, 0.0, False)
            self.prior_vao.render(moderngl.POINTS, vertices=self.prior_n)

        if self.overlay_n:
            # 1 px: a forward-compatible core context rejects glLineWidth > 1 (GL_INVALID_VALUE)
            self.overlay_vao.render(moderngl.LINES, vertices=self.overlay_n)

        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.screen.use()
        self.ctx.viewport = (0, 0, w, h)
        self.color_tex.use(0)
        self.depth_tex.use(1)
        e = self.edl_prog
        e["color_tex"].value = 0
        e["depth_tex"].value = 1
        e["texel"].value = (1.0 / w, 1.0 / h)
        e["strength"].value = float(opts.edl_strength)
        e["radius"].value = float(opts.edl_radius)
        e["near"].value = cam.near
        e["far"].value = cam.far
        e["enabled"].value = 1 if opts.edl else 0
        self.edl_vao.render(moderngl.TRIANGLES, vertices=3)
