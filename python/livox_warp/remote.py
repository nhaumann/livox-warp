"""Phone link: the viewer's cloud, pose and SLAM state in a phone's browser, and SLAM controls from it.

The viewer serves one page (remote_page.html) and a WebSocket on the same port. The page draws the cloud
itself with WebGL2, so orbiting stays smooth on the phone whatever the link does. Over the socket:

    viewer -> phone  binary  a cloud snapshot: CLOUD_HEADER, then int16 xyz in RES steps from an origin,
                             then RGB8. The points the viewer is showing, in its colours, thinned to the
                             phone's point budget.
                     text    JSON: "hello" once, then "status" STATUS_HZ times a second (pose, new path
                             points, odometry, prior map, rates).
    phone -> viewer  text    JSON: {"ack": seq} once a snapshot is on the phone's GPU, {"budget": points},
                             and commands {"cmd": ...} that the viewer runs on its render thread.

A client gets the next snapshot only after acknowledging the last one, so a slow link gets fewer
snapshots instead of a backlog. The WebSocket only opens with the link's key (?k=...); the URL carrying
the key is what the viewer shows as a QR code.

Threads: RemoteServer runs an asyncio loop on its own thread and hands commands over in a deque.
CloudPacker runs on the render thread: one kernel thins and quantises the shaded cloud, and the copy to
pinned memory is waited for with an event, never a stream sync.
"""

from __future__ import annotations

import asyncio
import collections
import hmac
import ipaddress
import json
import secrets
import socket
import struct
import threading
from dataclasses import dataclass
from http import HTTPStatus
from importlib import resources
from urllib.parse import parse_qs

import numpy as np
import segno
import warp as wp
from websockets.asyncio.server import ServerConnection, serve
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Response

from . import gpu

DEFAULT_PORT = 8765
STATUS_HZ = 10.0
CLOUD_MAX_HZ = 10.0
MAX_BITRATE = 40e6  # bits/s of snapshots at most: big ones go out less often, leaving the Wi-Fi room
MIN_BUDGET, DEFAULT_BUDGET, MAX_BUDGET = 20_000, 250_000, 1_000_000  # points per snapshot
MAX_CLIENTS = 4
MAX_PENDING_SENDS = 8  # a client this far behind (screen off, out of range) skips status messages

# Cloud snapshot: magic, sequence number, point count, origin xyz (m), metres per int16 step.
CLOUD_MAGIC = b"LWC1"
CLOUD_HEADER = struct.Struct("<4sII3ff")
RES = 0.01  # 1 cm steps: int16 reaches +-327 m around the origin, past the Mid-40's range
Q_MAX = wp.constant(32767.0)


@wp.kernel
def k_pack_cloud(
    pos: wp.array(dtype=wp.vec3),
    col: wp.array(dtype=wp.uint32),
    count: int,
    stride: int,
    origin: wp.vec3,
    inv_res: float,
    out_n: wp.array(dtype=wp.int32),
    out_pos: wp.array(dtype=wp.int16),
    out_col: wp.array(dtype=wp.uint8),
):
    """Every stride-th shaded point that is drawn and within int16 reach of the origin, in arbitrary order."""
    j = wp.tid() * stride
    if j >= count:
        return
    c = col[j]
    if (c >> wp.uint32(24)) == wp.uint32(0):
        return  # culled by k_shade (filters, hidden classes)
    q = (pos[j] - origin) * inv_res
    if wp.abs(q[0]) > Q_MAX or wp.abs(q[1]) > Q_MAX or wp.abs(q[2]) > Q_MAX:
        return
    k = wp.atomic_add(out_n, 0, 1)
    out_pos[3 * k + 0] = wp.int16(wp.round(q[0]))
    out_pos[3 * k + 1] = wp.int16(wp.round(q[1]))
    out_pos[3 * k + 2] = wp.int16(wp.round(q[2]))
    out_col[3 * k + 0] = wp.uint8(c & wp.uint32(255))
    out_col[3 * k + 1] = wp.uint8((c >> wp.uint32(8)) & wp.uint32(255))
    out_col[3 * k + 2] = wp.uint8((c >> wp.uint32(16)) & wp.uint32(255))


class CloudPacker:
    """Turns the shaded cloud into snapshot messages without stalling the render thread.

    pack() runs while the GL buffers are mapped (it reads the shaded positions and colours): it thins and
    quantises on the GPU, queues the copy into pinned memory on the render stream and records an event.
    poll() returns the message once that event has passed. One snapshot is in flight at a time.
    """

    def __init__(self, device, capacity: int = MAX_BUDGET):
        self.device = wp.get_device(device)
        d = self.device
        self.capacity = capacity
        self._n = wp.zeros(1, dtype=wp.int32, device=d)
        self._pos = wp.zeros(3 * capacity, dtype=wp.int16, device=d)
        self._col = wp.zeros(3 * capacity, dtype=wp.uint8, device=d)
        pinned = d.is_cuda
        self._h_n = wp.zeros(1, dtype=wp.int32, device="cpu", pinned=pinned)
        self._h_pos = wp.zeros(3 * capacity, dtype=wp.int16, device="cpu", pinned=pinned)
        self._h_col = wp.zeros(3 * capacity, dtype=wp.uint8, device="cpu", pinned=pinned)
        self._np_n, self._np_pos, self._np_col = self._h_n.numpy(), self._h_pos.numpy(), self._h_col.numpy()
        self._event = wp.Event(d) if d.is_cuda else None
        self._inflight = None  # (seq, origin) of the snapshot being copied
        self.seq = 0

    @property
    def busy(self) -> bool:
        return self._inflight is not None

    def pack(self, pos: wp.array, col: wp.array, count: int, origin, budget: int, stream=None) -> bool:
        """Start a snapshot of the first `count` shaded points, at most `budget` of them, around `origin`."""
        if self.busy or count <= 0:
            return False
        budget = max(1, min(int(budget), self.capacity))
        stride = -(-count // budget)
        slots = -(-count // stride)
        origin = np.round(np.asarray(origin, dtype=np.float64) / RES) * RES
        d = self.device
        wp.launch(gpu.k_fill_i32, dim=1, inputs=[self._n, 0], device=d)
        wp.launch(k_pack_cloud, dim=slots,
                  inputs=[pos, col, count, stride, wp.vec3(*origin.astype(np.float32)), 1.0 / RES],
                  outputs=[self._n, self._pos, self._col], device=d)
        if d.is_cuda:
            wp.copy(self._h_n, self._n, stream=stream)
            wp.copy(self._h_pos, self._pos, count=3 * slots, stream=stream)
            wp.copy(self._h_col, self._col, count=3 * slots, stream=stream)
            wp.record_event(self._event, stream)
        else:
            wp.copy(self._h_n, self._n)
            wp.copy(self._h_pos, self._pos, count=3 * slots)
            wp.copy(self._h_col, self._col, count=3 * slots)
        self.seq += 1
        self._inflight = (self.seq, origin)
        return True

    def poll(self) -> bytes | None:
        """The finished snapshot message, or None while it is still on its way (or none was started)."""
        if not self.busy or (self._event is not None and not self._event.is_complete):
            return None
        seq, origin = self._inflight
        self._inflight = None
        n = int(self._np_n[0])
        header = CLOUD_HEADER.pack(CLOUD_MAGIC, seq, n, *origin, RES)
        return b"".join((header, self._np_pos[:3 * n].tobytes(), self._np_col[:3 * n].tobytes()))


def snapshot_gap(nbytes: int) -> float:
    """Seconds between the start of a snapshot of `nbytes` and the next: CLOUD_MAX_HZ, or longer for big ones."""
    return max(1.0 / CLOUD_MAX_HZ, 8.0 * nbytes / MAX_BITRATE)


def decode_cloud(data: bytes):
    """(seq, origin, xyz float32 (n, 3), rgb uint8 (n, 3)) of a snapshot message; the page's decoder in Python."""
    magic, seq, n, ox, oy, oz, res = CLOUD_HEADER.unpack_from(data)
    if magic != CLOUD_MAGIC:
        raise ValueError(f"not a cloud snapshot: {magic!r}")
    at = CLOUD_HEADER.size
    q = np.frombuffer(data, np.int16, 3 * n, at).reshape(n, 3)
    rgb = np.frombuffer(data, np.uint8, 3 * n, at + 6 * n).reshape(n, 3)
    origin = np.array([ox, oy, oz])
    return seq, origin, (origin + q * res).astype(np.float32), rgb


def new_key() -> str:
    return secrets.token_urlsafe(9)


def lan_address() -> str:
    """This computer's address on its default route: the one a phone on the same Wi-Fi reaches."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("192.0.2.1", 9))  # UDP connect sends nothing; it only picks the outgoing interface
            return s.getsockname()[0]
        except OSError:
            return socket.gethostbyname(socket.gethostname())


def reachable(ip: str) -> bool:
    """Whether another device could reach this address: not loopback, link-local (169.254.x.x) or unspecified."""
    try:
        a = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return not (a.is_loopback or a.is_link_local or a.is_unspecified)


def phone_addresses(interface_ips) -> list[str]:
    """The addresses a phone could open the link on: the default route's first, then the other interfaces'."""
    out = []
    for ip in [lan_address(), *interface_ips]:
        if reachable(ip) and ip not in out:
            out.append(ip)
    return out


def qr_modules(text: str) -> list[list[bool]]:
    """QR code of `text` as rows of dark (True) modules, without the quiet zone."""
    return [[bool(m) for m in row] for row in segno.make(text, error="m", micro=False).matrix]


def _page() -> bytes:
    return resources.files(__package__).joinpath("remote_page.html").read_bytes()


def _response(status: HTTPStatus, body: bytes, content_type: str) -> Response:
    headers = Headers([("Content-Type", content_type), ("Content-Length", str(len(body))),
                       ("Cache-Control", "no-store"), ("Connection", "close")])
    return Response(status.value, status.phrase, headers, body)


@dataclass
class _Client:
    conn: ServerConnection
    name: str  # remote address, for the log
    ready: bool = True  # acknowledged its last snapshot
    budget: int = DEFAULT_BUDGET
    path_gen: int = -1  # which path its points belong to (see publish_status)
    path_n: int = 0  # path points it has
    sending: int = 0  # messages queued to it and not yet written


class RemoteServer:
    """The page and the WebSocket, on an asyncio loop of their own.

    Render thread: start(), want_cloud / budget / n_clients (plain reads), publish_cloud(),
    publish_status(), the `commands` deque ((message, client name) pairs to run), close().
    """

    def __init__(self, key: str, port: int = DEFAULT_PORT, bind: str = "0.0.0.0", address: str | None = None,
                 hello: dict | None = None, log=print):
        """address: the host the link's URL names (default: `bind`, or the default route's address when
        listening on all of them); it may be changed while serving, to another address of this computer."""
        self.key = key
        self.port = port
        self.bind = bind
        self.hello = json.dumps({"type": "hello", **(hello or {})})
        self.log = log  # log(text, level) with level "info" / "ok" / "warn"; called on the server thread
        self.page = _page()
        self.commands = collections.deque()
        self.want_cloud = False  # some client is waiting for a snapshot
        self.budget = DEFAULT_BUDGET  # the largest budget among the clients waiting
        self.n_clients = 0
        self.address = address or (bind if bind not in ("", "0.0.0.0") else lan_address())
        self._clients: dict[ServerConnection, _Client] = {}
        self._tasks = set()  # sends in flight; asyncio keeps only weak references to tasks
        self._loop = None
        self._stop = None
        self._thread = None
        self._started = threading.Event()
        self._error = None

    @property
    def url(self) -> str:
        """The link to open on the phone; the key travels in the fragment, which browsers never send."""
        return f"http://{self.address}:{self.port}/#k={self.key}"

    # ---- render thread ---------------------------------------------------------------------------

    def start(self, timeout: float = 5.0) -> None:
        """Bind and serve; raises OSError when the port cannot be bound."""
        self._thread = threading.Thread(target=asyncio.run, args=(self._serve(),), name="phone-link", daemon=True)
        self._thread.start()
        if not self._started.wait(timeout):
            raise OSError("the phone link did not start")
        if self._error is not None:
            raise self._error

    def close(self) -> None:
        if self._loop is not None and self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def publish_cloud(self, data: bytes) -> None:
        """Send a snapshot to every client waiting for one."""
        self._call(self._send_cloud, data)

    def publish_status(self, status: dict, path: list, path_gen: int) -> None:
        """Send `status` to every client, with the points of `path` (an append-only list of xyz) it lacks.
        A new `path_gen` means the path restarted: clients drop theirs and get it from the start."""
        self._call(self._send_status, status, path, len(path), path_gen)

    def _call(self, fn, *args):
        if self._loop is not None and not self._loop.is_closed():
            try:
                self._loop.call_soon_threadsafe(fn, *args)
            except RuntimeError:  # the loop closed between the check and the call
                pass

    # ---- server thread -----------------------------------------------------------------------------

    async def _serve(self):
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        try:
            async with serve(self._handle, self.bind, self.port, process_request=self._http, compression=None,
                             max_size=4096, ping_interval=5, ping_timeout=10) as server:
                self.port = server.sockets[0].getsockname()[1]  # the real one when asked for port 0
                self._started.set()
                await self._stop.wait()
        except OSError as e:
            self._error = e
            self._started.set()

    def _http(self, conn: ServerConnection, request):
        """Serve the page on /, let /ws through to the WebSocket handshake only with the key."""
        path, _, query = request.path.partition("?")
        if path == "/ws":
            key = parse_qs(query).get("k", [""])[0]
            if not hmac.compare_digest(key.encode(), self.key.encode()):
                return conn.respond(HTTPStatus.FORBIDDEN, "wrong or missing key: open the link the viewer shows\n")
            return None
        if path in ("/", "/index.html"):
            return _response(HTTPStatus.OK, self.page, "text/html; charset=utf-8")
        return conn.respond(HTTPStatus.NOT_FOUND, "not found\n")

    async def _handle(self, conn: ServerConnection):
        if len(self._clients) >= MAX_CLIENTS:
            await conn.close(1013, "too many phones connected")
            return
        host, port = conn.remote_address[:2]
        client = _Client(conn, f"{host}:{port}")
        self._clients[conn] = client
        self._refresh()
        self.log(f"phone link: {client.name} connected", "ok")
        try:
            await conn.send(self.hello)
            async for message in conn:
                if isinstance(message, str):
                    self._on_message(client, message)
        except ConnectionClosed:
            pass
        finally:
            del self._clients[conn]
            self._refresh()
            self.log(f"phone link: {client.name} disconnected", "info")

    def _on_message(self, client: _Client, text: str):
        try:
            msg = json.loads(text)
        except ValueError:
            return
        if not isinstance(msg, dict):
            return
        if "ack" in msg:
            client.ready = True
        if isinstance(msg.get("budget"), (int, float)):
            client.budget = int(min(max(msg["budget"], MIN_BUDGET), MAX_BUDGET))
        if isinstance(msg.get("cmd"), str):
            self.commands.append((msg, client.name))
        self._refresh()

    def _refresh(self):
        waiting = [c for c in self._clients.values() if c.ready]
        self.budget = max((c.budget for c in waiting), default=DEFAULT_BUDGET)
        self.want_cloud = bool(waiting)
        self.n_clients = len(self._clients)

    def _send_cloud(self, data: bytes):
        for c in self._clients.values():
            if c.ready:
                c.ready = False
                self._send(c, data)
        self._refresh()

    def _send_status(self, status: dict, path: list, n: int, gen: int):
        for c in self._clients.values():
            if c.sending > MAX_PENDING_SENDS:
                continue
            reset = c.path_gen != gen
            if reset:
                c.path_gen, c.path_n = gen, 0
            new = path[c.path_n:n]
            c.path_n = n
            pts = [[round(float(v), 3) for v in p] for p in new]
            self._send(c, json.dumps({"type": "status", **status, "path": {"reset": reset, "pts": pts}}))

    def _send(self, client: _Client, message):
        client.sending += 1
        task = asyncio.ensure_future(self._send_one(client, message))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    @staticmethod
    async def _send_one(client: _Client, message):
        try:
            await client.conn.send(message)
        except ConnectionClosed:
            pass
        finally:
            client.sending -= 1
