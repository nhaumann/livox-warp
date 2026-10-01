"""Phone link (remote.py): the GPU cloud packer, the WebSocket protocol, and the viewer driven by a client.

1. CloudPacker on a synthetic shaded cloud: culled points and points beyond int16 reach are left out, the
   point budget holds, positions come back within half a centimetre and colours exactly.
2. The addresses a phone could use leave out loopback and link-local ones. RemoteServer on a free port: the
   page is served, a wrong key is refused, a client gets hello and status messages with path deltas, gets a
   snapshot only after acknowledging the last one, and its commands queue.
3. The viewer itself (--sim walk with odometry, --remote 0, a config file of its own): a client sees the
   odometry frames climb and snapshots arrive, and its reset restarts the path and the frame count.
"""

import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

import common
import numpy as np
import warp as wp
from livox_warp.remote import CloudPacker, RemoteServer, decode_cloud, phone_addresses, reachable
from websockets.exceptions import InvalidStatus
from websockets.sync.client import connect

KEY = "check-remote-key"


def wait_for(cond, timeout: float = 5.0, what: str = "condition"):
    t_end = time.perf_counter() + timeout
    while not cond():
        if time.perf_counter() > t_end:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.01)


def packer(dev):
    n = 50_000
    rng = np.random.default_rng(7)
    pos = rng.uniform(-30.0, 30.0, (n, 3)).astype(np.float32)
    pos[:100] = (1000.0, 0.0, 0.0)  # beyond int16 reach of the origin: left out
    rgb = (np.arange(n, dtype=np.uint64) * 2654435761 & 0xFFFFFF).astype(np.uint32)  # unique: an odd multiplier
    col = rgb | np.uint32(0xFF000000)
    col[100:5100] = rgb[100:5100]  # alpha 0: culled by the shade kernel
    keep = np.ones(n, bool)
    keep[:5100] = False
    origin = np.array([1.234, -5.678, 0.5])
    wpos = wp.array(pos, dtype=wp.vec3, device=dev)
    wcol = wp.array(col, dtype=wp.uint32, device=dev)
    pk = CloudPacker(dev, capacity=100_000)

    def snapshot(budget):
        assert pk.pack(wpos, wcol, n, origin, budget)
        assert not pk.pack(wpos, wcol, n, origin, budget), "a second snapshot started while one is in flight"
        data = None

        def done():
            nonlocal data
            data = pk.poll()
            return data is not None

        wait_for(done, what="the snapshot's copy")
        return decode_cloud(data)

    _, org, xyz, got_rgb = snapshot(100_000)
    key = got_rgb.astype(np.uint32) @ np.array([1, 256, 65536], np.uint32)
    order = np.argsort(key)
    want = np.flatnonzero(keep)  # rgb is unique, so its sort order pairs the points up
    want = want[np.argsort(rgb[want])]
    assert len(key) == keep.sum(), f"{len(key)} points packed, {keep.sum()} drawable within reach"
    assert np.array_equal(key[order], rgb[want]), "colours differ"
    err = np.abs(xyz[order] - pos[want]).max()
    assert err < 0.0051, f"position error {err * 100:.2f} cm"
    assert np.allclose(org, np.round(origin, 2)), org
    print(f"packer   : {len(key):,} of {n:,} points packed (culled and out-of-reach left out), "
          f"max position error {err * 1000:.1f} mm")

    _, _, xyz, _ = snapshot(1000)
    assert 0 < len(xyz) <= 1000, len(xyz)
    print(f"           budget 1000: {len(xyz)} points (every 50th, culled ones left out)")


def addresses():
    assert not reachable("127.0.0.1") and not reachable("169.254.10.2") and not reachable("0.0.0.0")
    assert reachable("192.168.0.114") and reachable("10.1.2.3") and not reachable("not an ip")
    addrs = phone_addresses(["127.0.0.1", "169.254.1.1", "10.9.8.7", "10.9.8.7"])
    assert "10.9.8.7" in addrs and len(addrs) == len(set(addrs)) and all(reachable(a) for a in addrs), addrs
    print(f"addresses: a phone could use {', '.join(addrs)} (loopback and link-local left out)")


def server():
    srv = RemoteServer(KEY, port=0, bind="127.0.0.1", hello={"colors": ["a", "b"]}, log=lambda text, level: None)
    srv.start()
    base = f"127.0.0.1:{srv.port}"
    try:
        with urllib.request.urlopen(f"http://{base}/") as r:
            assert r.headers["Content-Type"].startswith("text/html")
            assert b"<canvas" in r.read()
        try:
            connect(f"ws://{base}/ws?k=wrong", open_timeout=5).close()
            raise AssertionError("a wrong key was let in")
        except InvalidStatus as e:
            assert e.response.status_code == 403, e.response.status_code

        with connect(f"ws://{base}/ws?k={KEY}", open_timeout=5) as ws:
            hello = json.loads(ws.recv(timeout=5))
            assert hello == {"type": "hello", "colors": ["a", "b"]}, hello
            wait_for(lambda: srv.n_clients == 1 and srv.want_cloud, what="the client to count")
            ws.send(json.dumps({"budget": 123_456}))
            wait_for(lambda: srv.budget == 123_456, what="the budget")

            def status(path, gen):
                srv.publish_status({"odom": None}, path, gen)
                return json.loads(ws.recv(timeout=5))["path"]

            path = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])]
            assert status(path, 7) == {"reset": True, "pts": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]}
            path.append(np.array([2.0, 0.0, 0.0]))
            assert status(path, 7) == {"reset": False, "pts": [[2.0, 0.0, 0.0]]}
            assert status([np.array([5.0, 5.0, 5.0])], 8) == {"reset": True, "pts": [[5.0, 5.0, 5.0]]}

            srv.publish_cloud(b"one")
            assert ws.recv(timeout=5) == b"one"
            wait_for(lambda: not srv.want_cloud, what="the client to wait for its ack")
            srv.publish_cloud(b"two")  # not acknowledged yet: not sent
            try:
                ws.recv(timeout=0.5)
                raise AssertionError("a snapshot went out before the last one was acknowledged")
            except TimeoutError:
                pass
            ws.send(json.dumps({"ack": 1}))
            wait_for(lambda: srv.want_cloud, what="the ack")
            srv.publish_cloud(b"three")
            assert ws.recv(timeout=5) == b"three"

            ws.send(json.dumps({"cmd": "reset"}))
            wait_for(lambda: len(srv.commands) == 1, what="the command")
            msg, who = srv.commands.popleft()
            assert msg == {"cmd": "reset"} and who.startswith("127.0.0.1:"), (msg, who)
        wait_for(lambda: srv.n_clients == 0, what="the client to leave")
        print(f"server   : page, key check, hello, path deltas, ack-gated snapshots, commands on port {srv.port}")
    finally:
        srv.close()


def viewer():
    """The real viewer on the simulator, driven by a client as the phone page would."""
    with tempfile.TemporaryDirectory() as tmp:
        python_path = os.pathsep.join([str(common.ROOT / "python"), os.environ.get("PYTHONPATH", "")])
        env = dict(os.environ, PYTHONPATH=python_path)
        cmd = [sys.executable, "-u", "-m", "livox_warp", "--sim", "--sim-motion", "walk", "--odom", "--remote", "0",
               "--remote-bind", "127.0.0.1", "--config", os.path.join(tmp, "viewer.json"), "--size", "800x600"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
                                encoding="utf-8", errors="replace")
        lines = queue.Queue()
        threading.Thread(target=lambda: [lines.put(line) for line in proc.stdout], daemon=True).start()
        try:
            url = None
            t_end = time.perf_counter() + 180.0
            while url is None:
                try:
                    line = lines.get(timeout=max(0.0, t_end - time.perf_counter()))
                except queue.Empty:
                    raise AssertionError("the viewer never started its phone link") from None
                m = re.search(r"phone link: open http://(\S+?)/#k=(\S+)", line)
                url = m and (m.group(1), m.group(2))
            host, key = url
            with connect(f"ws://{host}/ws?k={key}", open_timeout=10) as ws:
                state = {"snapshots": 0, "points": 0, "frames": 0, "path": 0, "color": None, "reset_seen": False}

                def pump(seconds: float, until=None):
                    t_stop = time.perf_counter() + seconds
                    while time.perf_counter() < t_stop and not (until and until()):
                        try:
                            msg = ws.recv(timeout=0.5)
                        except TimeoutError:
                            continue
                        if isinstance(msg, bytes):
                            seq, _, xyz, rgb = decode_cloud(msg)
                            state["snapshots"] += 1
                            state["points"] = len(xyz)
                            assert len(xyz) == 0 or rgb.any(), "a snapshot with all-black points"
                            ws.send(json.dumps({"ack": seq}))
                            continue
                        st = json.loads(msg)
                        if st["type"] != "status":
                            continue
                        if st["path"]["reset"]:
                            state["path"] = 0
                            state["reset_seen"] = True
                        state["path"] += len(st["path"]["pts"])
                        state["frames"] = st["odom"]["frames"] if st["odom"] else 0
                        state["color"] = st["color"]
                        assert st["sampling"] is None, "the simulator reported LiDAR sampling"
                        assert st["recording"] is None, "the simulator offered recording"

                pump(60.0, lambda: state["frames"] >= 40 and state["snapshots"] >= 5)
                assert state["frames"] >= 40, f"odometry frames stuck at {state['frames']}"
                assert state["snapshots"] >= 5 and state["points"] > 1000, state
                before = dict(state)
                print(f"viewer   : {before['snapshots']} snapshots (last {before['points']:,} points), "
                      f"odometry at {before['frames']} frames, path {before['path']} points")

                state["reset_seen"] = False
                ws.send(json.dumps({"cmd": "reset"}))
                ws.send(json.dumps({"cmd": "color", "mode": "height"}))
                pump(10.0, lambda: state["reset_seen"] and state["color"] == "height")
                assert state["reset_seen"], "the reset never restarted the path"
                pump(1.0)
                assert state["frames"] < before["frames"], (state["frames"], before["frames"])
                assert state["color"] == "height", state["color"]
                print(f"           after a reset from the client: odometry at {state['frames']} frames, "
                      f"path {state['path']} points, colours {state['color']}")
        finally:
            proc.terminate()
            proc.wait(timeout=30)


def main():
    dev = common.device()
    packer(dev)
    addresses()
    server()
    viewer()
    print("OK")


if __name__ == "__main__":
    main()
