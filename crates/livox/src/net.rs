//! Discovery, device sessions (handshake + heartbeat + commands) and point reception.
//!
//! Unlike Livox Viewer, the host address is an explicit input: the caller chooses which local
//! IPv4 the LiDAR streams to, so multi-homed or multi-address adapters are never ambiguous.

use crate::point::{self, DataHeader, Imu, StreamProfile};
use crate::proto::{
    self, cmd_set, general, lidar, ptype, DeviceType, Extrinsic, Heartbeat, IpInfo,
};
use crate::record;
use socket2::{Domain, Protocol, Socket, Type};
use std::fmt;
use std::io;
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4, UdpSocket};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU16, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// Read timeout on every socket, so receive loops notice a stop request promptly.
const POLL: Duration = Duration::from_millis(200);
/// Pause after a socket error that is not a timeout, so a persistent one cannot spin a thread.
const SOCKET_ERROR_BACKOFF: Duration = Duration::from_millis(10);
/// Kernel receive buffer asked for on the data socket; the default is too small for a point stream.
const DATA_RECV_BUFFER: usize = 16 << 20;
/// A timestamp jump this large between consecutive packets is a clock change, not packet loss.
const CLOCK_JUMP_NS: u64 = 1_000_000_000;

/// SDK-default command port for the first device (the SDK uses 55500 + n).
pub const DEFAULT_CMD_PORT: u16 = 55501;
/// SDK-default point-data port for the first device (the SDK uses 56000 + n).
pub const DEFAULT_DATA_PORT: u16 = 56001;
/// Points a [`Sink`] holds between drains before it drops new ones: several seconds of any SDK1
/// sensor's output, even an Avia in triple-return mode.
pub const DEFAULT_MAX_BUFFERED_POINTS: usize = 4_000_000;

/// Locks `m`, taking the data back if another thread panicked while holding it. Every critical
/// section in this module leaves its state usable if interrupted (at worst a stale batch or
/// status), so one crashed thread should not turn every later call, e.g. from Python, into a
/// panic as well.
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

// ---------------------------------------------------------------------------------------------
// Local interfaces
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalIface {
    pub name: String,
    pub ip: Ipv4Addr,
    pub mask: Ipv4Addr,
}

impl LocalIface {
    pub fn contains(&self, ip: Ipv4Addr) -> bool {
        same_subnet(self.ip, ip, self.mask)
    }
}

pub fn same_subnet(a: Ipv4Addr, b: Ipv4Addr, mask: Ipv4Addr) -> bool {
    let m = u32::from(mask);
    u32::from(a) & m == u32::from(b) & m
}

/// Every non-loopback, non-link-local IPv4 address on this machine (one entry per address,
/// so an adapter carrying two addresses appears twice).
pub fn local_ipv4() -> Vec<LocalIface> {
    let Ok(ifaces) = if_addrs::get_if_addrs() else {
        return Vec::new();
    };
    ifaces
        .into_iter()
        .filter_map(|i| match i.addr {
            if_addrs::IfAddr::V4(v4) if !v4.ip.is_loopback() && !v4.ip.is_link_local() => {
                Some(LocalIface {
                    name: i.name,
                    ip: v4.ip,
                    mask: v4.netmask,
                })
            }
            _ => None,
        })
        .collect()
}

/// The local address on the same subnet as `lidar`, if there is one.
pub fn host_for(lidar: Ipv4Addr) -> Option<LocalIface> {
    local_ipv4().into_iter().find(|i| i.contains(lidar))
}

// ---------------------------------------------------------------------------------------------
// Sockets
// ---------------------------------------------------------------------------------------------

#[derive(Default)]
struct SocketOptions {
    /// `SO_REUSEADDR`. Only the discovery listener wants it, to share UDP 55000 with Livox
    /// Viewer. Command and data sockets stay exclusive, so a second instance fails to bind
    /// instead of silently splitting the point stream with the first.
    reuse_address: bool,
    broadcast: bool,
    /// Kernel receive buffer to ask for; 0 keeps the system default.
    recv_buffer: usize,
}

fn udp_socket(bind: SocketAddrV4, opts: SocketOptions) -> io::Result<UdpSocket> {
    let s = Socket::new(Domain::IPV4, Type::DGRAM, Some(Protocol::UDP))?;
    if opts.reuse_address {
        s.set_reuse_address(true)?;
    }
    if opts.broadcast {
        s.set_broadcast(true)?;
    }
    if opts.recv_buffer > 0 {
        // Best effort: the kernel clamps the request to its own limit.
        let _ = s.set_recv_buffer_size(opts.recv_buffer);
    }
    s.bind(&SocketAddr::V4(bind).into())?;
    let u: UdpSocket = s.into();
    u.set_read_timeout(Some(POLL))?;
    Ok(u)
}

fn is_timeout(e: &io::Error) -> bool {
    matches!(
        e.kind(),
        io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
    )
}

/// One receive on a socket with a read timeout: `None` after a timeout, or after a short pause
/// on any other error. Windows reports an ICMP port-unreachable as a connection reset on the
/// next receive, and an interface going away must not spin the thread.
fn recv_or_backoff(sock: &UdpSocket, buf: &mut [u8]) -> Option<(usize, SocketAddr)> {
    match sock.recv_from(buf) {
        Ok(v) => Some(v),
        Err(e) if is_timeout(&e) => None,
        Err(_) => {
            thread::sleep(SOCKET_ERROR_BACKOFF);
            None
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Discovery
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Seen {
    pub code: String,
    pub dev_type: DeviceType,
    pub ip: Ipv4Addr,
    pub last_seen: Instant,
    pub broadcasts: u64,
}

struct DiscoveryShared {
    seen: Mutex<Vec<Seen>>,
    running: AtomicBool,
    datagrams: AtomicU64,
    bad_frames: AtomicU64,
}

/// Listens for LiDAR broadcasts on UDP 55000. A LiDAR broadcasts about once a second until a
/// host completes a handshake with it.
pub struct Discovery {
    shared: Arc<DiscoveryShared>,
    thread: Option<JoinHandle<()>>,
}

impl Discovery {
    pub fn start() -> io::Result<Self> {
        let sock = udp_socket(
            SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, proto::BROADCAST_PORT),
            SocketOptions {
                reuse_address: true,
                broadcast: true,
                recv_buffer: 0,
            },
        )?;
        let shared = Arc::new(DiscoveryShared {
            seen: Mutex::new(Vec::new()),
            running: AtomicBool::new(true),
            datagrams: AtomicU64::new(0),
            bad_frames: AtomicU64::new(0),
        });
        let sh = shared.clone();
        let thread = thread::Builder::new()
            .name("livox-discovery".into())
            .spawn(move || {
                let mut buf = [0u8; 1500];
                while sh.running.load(Ordering::Relaxed) {
                    let Some((n, from)) = recv_or_backoff(&sock, &mut buf) else {
                        continue;
                    };
                    sh.datagrams.fetch_add(1, Ordering::Relaxed);
                    let SocketAddr::V4(from) = from else { continue };
                    let Ok(frame) = proto::decode(&buf[..n]) else {
                        sh.bad_frames.fetch_add(1, Ordering::Relaxed);
                        continue;
                    };
                    let Some(b) = proto::parse_broadcast(&frame) else {
                        continue;
                    };
                    let mut seen = lock(&sh.seen);
                    match seen.iter_mut().find(|s| s.code == b.code) {
                        Some(s) => {
                            s.ip = *from.ip();
                            s.dev_type = b.dev_type;
                            s.last_seen = Instant::now();
                            s.broadcasts += 1;
                        }
                        None => seen.push(Seen {
                            code: b.code,
                            dev_type: b.dev_type,
                            ip: *from.ip(),
                            last_seen: Instant::now(),
                            broadcasts: 1,
                        }),
                    }
                }
            })?;
        Ok(Self {
            shared,
            thread: Some(thread),
        })
    }

    /// LiDARs heard within `max_age`.
    pub fn devices(&self, max_age: Duration) -> Vec<Seen> {
        let seen = lock(&self.shared.seen);
        seen.iter()
            .filter(|s| s.last_seen.elapsed() <= max_age)
            .cloned()
            .collect()
    }

    /// Datagrams received on 55000 (valid or not). Zero after a few seconds usually means
    /// another program (Livox Viewer) has the port bound to a specific address.
    pub fn datagrams(&self) -> u64 {
        self.shared.datagrams.load(Ordering::Relaxed)
    }

    pub fn bad_frames(&self) -> u64 {
        self.shared.bad_frames.load(Ordering::Relaxed)
    }
}

impl Drop for Discovery {
    fn drop(&mut self) {
        self.shared.running.store(false, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Point sink (shared by live devices and replays)
// ---------------------------------------------------------------------------------------------

/// Points in structure-of-arrays form, ready to hand to the GPU.
#[derive(Debug, Default, Clone)]
pub struct PointBatch {
    /// x, y, z in metres, interleaved.
    pub xyz: Vec<f32>,
    /// reflectivity | tag << 8 | return << 16 | device index << 24
    pub attr: Vec<u32>,
    /// Seconds since the session's first packet, on the LiDAR's clock, as f32: that resolves
    /// about 4 us (under one firing interval) during the first minute, 0.25 ms after an hour
    /// and 1 ms after four. Fine for ordering points and compensating motion within a frame;
    /// not an absolute timestamp.
    pub t: Vec<f32>,
}

impl PointBatch {
    pub fn len(&self) -> usize {
        self.t.len()
    }
    pub fn is_empty(&self) -> bool {
        self.t.is_empty()
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Stats {
    pub packets: u64,
    pub points: u64,
    pub bytes: u64,
    pub lost_packets: u64,
    pub bad_packets: u64,
    pub imu_samples: u64,
    pub dropped_points: u64,
    pub data_type: u8,
    pub data_status: u32,
    pub timestamp_type: u8,
}

#[derive(Default)]
struct SinkState {
    profile: StreamProfile,
    batch: PointBatch,
    t_base: Option<u64>,
    expected_next: Option<u64>,
    imu: Option<Imu>,
    stats: Stats,
}

/// Turns raw data-port datagrams into a [`PointBatch`], tracking loss and the time base.
pub struct Sink {
    dev_type: DeviceType,
    dev_index: u8,
    max_points: usize,
    state: Mutex<SinkState>,
}

impl Sink {
    pub fn new(dev_type: DeviceType, dev_index: u8, max_points: usize) -> Self {
        Self {
            dev_type,
            dev_index,
            max_points,
            state: Mutex::new(SinkState::default()),
        }
    }

    pub fn set_profile(&self, profile: StreamProfile) {
        let mut st = lock(&self.state);
        st.profile = profile;
        st.batch = PointBatch::default();
        st.t_base = None;
        st.expected_next = None;
    }

    pub fn profile(&self) -> StreamProfile {
        lock(&self.state).profile
    }

    pub fn ingest(&self, pkt: &[u8]) {
        let mut st = lock(&self.state);
        let Some(h) = DataHeader::parse(pkt) else {
            st.stats.bad_packets += 1;
            return;
        };
        st.stats.packets += 1;
        st.stats.bytes += pkt.len() as u64;
        st.stats.data_type = h.data_type;
        st.stats.data_status = h.status;
        st.stats.timestamp_type = h.timestamp_type;

        if h.data_type == point::data_type::IMU {
            if let Some(imu) = point::parse_imu(pkt) {
                st.imu = Some(imu);
                st.stats.imu_samples += 1;
            }
            return;
        }

        let ts = h.timestamp_ns();
        let interval = self.dev_type.sample_interval_ns();
        let profile = st.profile;
        let records = point::records_in(pkt, h.data_type) as u64;
        let span = if profile == StreamProfile::Mid40Dual {
            records / 2
        } else {
            records
        } * interval;
        let mut base = match st.t_base {
            Some(b) if ts >= b => b,
            _ => {
                // First packet, or the LiDAR clock went backwards (reboot / sync change).
                st.t_base = Some(ts);
                st.expected_next = None;
                ts
            }
        };
        if let (Some(exp), true) = (st.expected_next, span > 0) {
            if ts > exp + span / 2 {
                let gap = ts - exp;
                if gap < CLOCK_JUMP_NS {
                    st.stats.lost_packets += (gap + span / 2) / span;
                } else {
                    // The clock moved, e.g. the Mid-40's first packet after a handshake carries
                    // a stale timestamp. Shift the base so t stays continuous.
                    base += gap;
                    st.t_base = Some(base);
                }
            }
        }
        st.expected_next = Some(ts + span);

        let tag_dev = (self.dev_index as u32) << 24;
        let mut added = 0u64;
        let mut dropped = 0u64;
        let SinkState { batch, .. } = &mut *st;
        let parsed = point::for_each_point_profile(pkt, profile, |p| {
            if batch.len() >= self.max_points {
                dropped += 1;
                return;
            }
            batch.xyz.extend_from_slice(&[p.x, p.y, p.z]);
            batch
                .attr
                .push(p.reflectivity as u32 | (p.tag as u32) << 8 | (p.ret as u32) << 16 | tag_dev);
            batch
                .t
                .push((((ts - base) + p.record as u64 * interval) as f64 * 1e-9) as f32);
            added += 1;
        });
        if parsed.is_none() {
            st.stats.bad_packets += 1;
        }
        st.stats.points += added;
        st.stats.dropped_points += dropped;
    }

    /// Everything received since the last drain.
    pub fn drain(&self) -> PointBatch {
        std::mem::take(&mut lock(&self.state).batch)
    }

    pub fn stats(&self) -> Stats {
        lock(&self.state).stats
    }

    pub fn imu(&self) -> Option<Imu> {
        lock(&self.state).imu
    }

    /// Forget the time base, e.g. when a replay loops.
    pub fn reset_clock(&self) {
        let mut st = lock(&self.state);
        st.t_base = None;
        st.expected_next = None;
    }
}

// ---------------------------------------------------------------------------------------------
// Device session
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ConnectConfig {
    pub lidar_ip: Ipv4Addr,
    /// The local address the LiDAR will send acks and points to. Must be on the LiDAR's subnet.
    pub host_ip: Ipv4Addr,
    pub dev_type: DeviceType,
    pub cmd_port: u16,
    pub data_port: u16,
    /// Where IMU samples arrive. Equal to `data_port` by default, so they share the data
    /// socket; a different port gets a socket of its own.
    pub imu_port: u16,
    /// Stored in the top byte of every point's attr word, to tell LiDARs apart.
    pub dev_index: u8,
    /// Points held between drains before new ones are dropped.
    pub max_buffered_points: usize,
}

impl ConnectConfig {
    /// SDK-default ports for the first device and [`DEFAULT_MAX_BUFFERED_POINTS`].
    pub fn new(lidar_ip: Ipv4Addr, host_ip: Ipv4Addr, dev_type: DeviceType) -> Self {
        Self {
            lidar_ip,
            host_ip,
            dev_type,
            cmd_port: DEFAULT_CMD_PORT,
            data_port: DEFAULT_DATA_PORT,
            imu_port: DEFAULT_DATA_PORT,
            dev_index: 0,
            max_buffered_points: DEFAULT_MAX_BUFFERED_POINTS,
        }
    }
}

#[derive(Debug)]
pub enum CmdError {
    Io(io::Error),
    Timeout,
    Rejected(u8),
    BadReply,
    NotConnected,
}

impl fmt::Display for CmdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CmdError::Io(e) => write!(f, "socket error: {e}"),
            CmdError::Timeout => f.write_str("no reply from the LiDAR"),
            CmdError::Rejected(code) => write!(f, "LiDAR rejected the command (ret_code {code})"),
            CmdError::BadReply => f.write_str("malformed reply"),
            CmdError::NotConnected => f.write_str("not connected"),
        }
    }
}

impl std::error::Error for CmdError {}

impl From<io::Error> for CmdError {
    fn from(e: io::Error) -> Self {
        CmdError::Io(e)
    }
}

#[derive(Debug, Clone, Default)]
pub struct Status {
    pub connected: bool,
    pub heartbeat: Option<Heartbeat>,
    pub heartbeat_age: Option<Duration>,
    pub missed_heartbeats: u32,
    pub firmware: Option<[u8; 4]>,
    /// Last pushed abnormal-status word (general 0x07 MSG), if any.
    pub abnormal: Option<u32>,
}

#[derive(Default)]
struct StatusState {
    heartbeat: Option<Heartbeat>,
    last_heartbeat: Option<Instant>,
    missed: u32,
    firmware: Option<[u8; 4]>,
    abnormal: Option<u32>,
}

/// An ACK frame, as the command thread hands it to the request waiting for it.
struct Ack {
    seq: u16,
    cmd_set: u8,
    cmd_id: u8,
    payload: Vec<u8>,
}

struct Inner {
    cfg: ConnectConfig,
    cmd_sock: UdpSocket,
    lidar: SocketAddr,
    seq: AtomicU16,
    /// Serialises requests: the LiDAR answers one command at a time.
    cmd_lock: Mutex<()>,
    ack: Mutex<Option<Ack>>,
    ack_cv: Condvar,
    status: Mutex<StatusState>,
    sink: Arc<Sink>,
    running: AtomicBool,
    connected: AtomicBool,
    recorder: Mutex<Option<record::Writer>>,
    data_datagrams: AtomicU64,
}

impl Inner {
    fn request_typed(
        &self,
        ty: u8,
        set: u8,
        id: u8,
        payload: &[u8],
        timeout: Duration,
    ) -> Result<Vec<u8>, CmdError> {
        let _serial = lock(&self.cmd_lock);
        *lock(&self.ack) = None;
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        self.cmd_sock
            .send_to(&proto::encode(ty, seq, set, id, payload), self.lidar)?;
        let deadline = Instant::now() + timeout;
        let mut guard = lock(&self.ack);
        loop {
            if let Some(ack) = guard.take() {
                if (ack.seq, ack.cmd_set, ack.cmd_id) == (seq, set, id) {
                    return match ack.payload.split_first() {
                        None => Err(CmdError::BadReply),
                        Some((0, body)) => Ok(body.to_vec()),
                        Some((&code, _)) => Err(CmdError::Rejected(code)),
                    };
                }
            }
            let now = Instant::now();
            if now >= deadline {
                return Err(CmdError::Timeout);
            }
            guard = self
                .ack_cv
                .wait_timeout(guard, deadline - now)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
    }

    fn request(&self, set: u8, id: u8, payload: &[u8]) -> Result<Vec<u8>, CmdError> {
        self.request_typed(ptype::CMD, set, id, payload, Duration::from_millis(1000))
    }
}

/// A connected LiDAR. Dropping it sends a disconnect and stops its threads.
pub struct Device {
    inner: Arc<Inner>,
    threads: Mutex<Vec<JoinHandle<()>>>,
}

impl Device {
    pub fn connect(cfg: ConnectConfig) -> Result<Self, CmdError> {
        let cmd_sock = udp_socket(
            SocketAddrV4::new(cfg.host_ip, cfg.cmd_port),
            SocketOptions::default(),
        )?;
        let data_sock = udp_socket(
            SocketAddrV4::new(cfg.host_ip, cfg.data_port),
            SocketOptions {
                recv_buffer: DATA_RECV_BUFFER,
                ..SocketOptions::default()
            },
        )?;
        let imu_sock = if cfg.imu_port == cfg.data_port {
            None
        } else {
            Some(udp_socket(
                SocketAddrV4::new(cfg.host_ip, cfg.imu_port),
                SocketOptions::default(),
            )?)
        };
        let inner = Arc::new(Inner {
            lidar: SocketAddr::V4(SocketAddrV4::new(cfg.lidar_ip, proto::LIDAR_CMD_PORT)),
            sink: Arc::new(Sink::new(
                cfg.dev_type,
                cfg.dev_index,
                cfg.max_buffered_points,
            )),
            cfg,
            cmd_sock,
            seq: AtomicU16::new(1),
            cmd_lock: Mutex::new(()),
            ack: Mutex::new(None),
            ack_cv: Condvar::new(),
            status: Mutex::new(StatusState::default()),
            running: AtomicBool::new(true),
            connected: AtomicBool::new(false),
            recorder: Mutex::new(None),
            data_datagrams: AtomicU64::new(0),
        });
        let dev = Self {
            inner: inner.clone(),
            threads: Mutex::new(Vec::new()),
        };
        dev.spawn("livox-cmd", Self::cmd_loop)?;
        dev.spawn("livox-data", move |inner| Self::data_loop(inner, data_sock))?;
        if let Some(sock) = imu_sock {
            dev.spawn("livox-imu", move |inner| Self::data_loop(inner, sock))?;
        }

        // The SDK sends the handshake with packet type ACK (device_discovery.cpp); fall back to
        // CMD in case a firmware is stricter.
        let hs = proto::handshake_payload(
            inner.cfg.host_ip,
            inner.cfg.data_port,
            inner.cfg.cmd_port,
            inner.cfg.imu_port,
        );
        let mut result = Err(CmdError::Timeout);
        for ty in [ptype::ACK, ptype::ACK, ptype::CMD, ptype::CMD] {
            result = inner.request_typed(
                ty,
                cmd_set::GENERAL,
                general::HANDSHAKE,
                &hs,
                Duration::from_millis(700),
            );
            if result.is_ok() {
                break;
            }
        }
        result?;
        inner.connected.store(true, Ordering::Relaxed);
        lock(&inner.status).last_heartbeat = Some(Instant::now());
        dev.spawn("livox-heartbeat", Self::heartbeat_loop)?;

        if let Ok([a, b, c, d, ..]) = inner
            .request(cmd_set::GENERAL, general::DEVICE_INFO, &[])
            .as_deref()
        {
            let firmware = [*a, *b, *c, *d];
            lock(&inner.status).firmware = Some(firmware);
            inner
                .sink
                .set_profile(StreamProfile::from_firmware(inner.cfg.dev_type, firmware));
        }
        // Cartesian output; the Warp side expects xyz.
        let _ = inner.request(cmd_set::GENERAL, general::COORDINATE, &[0]);
        Ok(dev)
    }

    fn spawn(&self, name: &str, f: impl FnOnce(Arc<Inner>) + Send + 'static) -> io::Result<()> {
        let inner = self.inner.clone();
        let t = thread::Builder::new()
            .name(name.into())
            .spawn(move || f(inner))?;
        lock(&self.threads).push(t);
        Ok(())
    }

    fn cmd_loop(inner: Arc<Inner>) {
        let mut buf = [0u8; 1500];
        while inner.running.load(Ordering::Relaxed) {
            let Some((n, from)) = recv_or_backoff(&inner.cmd_sock, &mut buf) else {
                continue;
            };
            if from.ip() != inner.lidar.ip() {
                continue;
            }
            let Ok(f) = proto::decode(&buf[..n]) else {
                continue;
            };
            match f.ptype {
                ptype::ACK => {
                    *lock(&inner.ack) = Some(Ack {
                        seq: f.seq,
                        cmd_set: f.cmd_set,
                        cmd_id: f.cmd_id,
                        payload: f.payload.to_vec(),
                    });
                    inner.ack_cv.notify_all();
                }
                ptype::MSG => {
                    if let Some(word) = proto::parse_abnormal_status(&f) {
                        lock(&inner.status).abnormal = Some(word);
                    }
                }
                _ => {}
            }
        }
    }

    /// Receives point (or IMU) datagrams on `sock` into the sink, recording them if asked.
    fn data_loop(inner: Arc<Inner>, sock: UdpSocket) {
        let mut buf = [0u8; 2048];
        while inner.running.load(Ordering::Relaxed) {
            let Some((n, _)) = recv_or_backoff(&sock, &mut buf) else {
                continue;
            };
            inner.data_datagrams.fetch_add(1, Ordering::Relaxed);
            let pkt = &buf[..n];
            if let Some(w) = lock(&inner.recorder).as_mut() {
                let _ = w.write(pkt);
            }
            inner.sink.ingest(pkt);
        }
    }

    fn heartbeat_loop(inner: Arc<Inner>) {
        let mut next = Instant::now();
        while inner.running.load(Ordering::Relaxed) {
            next += Duration::from_secs(1);
            let reply = inner.request_typed(
                ptype::CMD,
                cmd_set::GENERAL,
                general::HEARTBEAT,
                &[],
                Duration::from_millis(600),
            );
            {
                let mut st = lock(&inner.status);
                match reply.ok().and_then(|p| Heartbeat::parse(&p)) {
                    Some(hb) => {
                        st.heartbeat = Some(hb);
                        st.last_heartbeat = Some(Instant::now());
                        st.missed = 0;
                        inner.connected.store(true, Ordering::Relaxed);
                    }
                    None => {
                        st.missed += 1;
                        // The LiDAR drops the session after ~3 s without heartbeats.
                        if st.missed >= 3 {
                            inner.connected.store(false, Ordering::Relaxed);
                        }
                    }
                }
            }
            while inner.running.load(Ordering::Relaxed) && Instant::now() < next {
                thread::sleep(Duration::from_millis(50));
            }
        }
    }

    pub fn config(&self) -> &ConnectConfig {
        &self.inner.cfg
    }

    pub fn stream_profile(&self) -> StreamProfile {
        self.inner.sink.profile()
    }

    pub fn status(&self) -> Status {
        let st = lock(&self.inner.status);
        Status {
            connected: self.inner.connected.load(Ordering::Relaxed),
            heartbeat: st.heartbeat,
            heartbeat_age: st.last_heartbeat.map(|t| t.elapsed()),
            missed_heartbeats: st.missed,
            firmware: st.firmware,
            abnormal: st.abnormal,
        }
    }

    pub fn stats(&self) -> Stats {
        self.inner.sink.stats()
    }

    pub fn data_datagrams(&self) -> u64 {
        self.inner.data_datagrams.load(Ordering::Relaxed)
    }

    pub fn drain(&self) -> PointBatch {
        self.inner.sink.drain()
    }

    pub fn imu(&self) -> Option<Imu> {
        self.inner.sink.imu()
    }

    fn cmd(&self, set: u8, id: u8, payload: &[u8]) -> Result<Vec<u8>, CmdError> {
        if !self.inner.running.load(Ordering::Relaxed) {
            return Err(CmdError::NotConnected);
        }
        self.inner.request(set, id, payload)
    }

    pub fn start_sampling(&self) -> Result<(), CmdError> {
        self.cmd(cmd_set::GENERAL, general::SAMPLING, &[1])
            .map(drop)
    }

    pub fn stop_sampling(&self) -> Result<(), CmdError> {
        self.cmd(cmd_set::GENERAL, general::SAMPLING, &[0])
            .map(drop)
    }

    pub fn ip_info(&self) -> Result<IpInfo, CmdError> {
        let p = self.cmd(cmd_set::GENERAL, general::GET_IP, &[])?;
        IpInfo::parse(&p).ok_or(CmdError::BadReply)
    }

    /// Takes effect after the LiDAR reboots (see [`Device::reboot`]).
    pub fn set_ip(&self, ip: &IpInfo) -> Result<(), CmdError> {
        self.cmd(
            cmd_set::GENERAL,
            general::SET_IP,
            &proto::set_ip_payload(ip),
        )
        .map(drop)
    }

    pub fn reboot(&self, delay_ms: u16) -> Result<(), CmdError> {
        self.cmd(cmd_set::GENERAL, general::REBOOT, &delay_ms.to_le_bytes())
            .map(drop)
    }

    /// 1 normal, 2 power-saving, 3 standby.
    pub fn set_mode(&self, mode: u8) -> Result<(), CmdError> {
        self.cmd(cmd_set::LIDAR, lidar::SET_MODE, &[mode]).map(drop)
    }

    /// 0 single first, 1 single strongest, 2 dual, 3 triple (Avia).
    pub fn set_return_mode(&self, mode: u8) -> Result<(), CmdError> {
        self.cmd(cmd_set::LIDAR, lidar::SET_RETURN_MODE, &[mode])
            .map(drop)
    }

    pub fn return_mode(&self) -> Result<u8, CmdError> {
        self.cmd(cmd_set::LIDAR, lidar::GET_RETURN_MODE, &[])?
            .first()
            .copied()
            .ok_or(CmdError::BadReply)
    }

    pub fn set_fan(&self, on: bool) -> Result<(), CmdError> {
        self.cmd(cmd_set::LIDAR, lidar::SET_FAN, &[on as u8])
            .map(drop)
    }

    pub fn fan(&self) -> Result<bool, CmdError> {
        Ok(self
            .cmd(cmd_set::LIDAR, lidar::GET_FAN, &[])?
            .first()
            .copied()
            .ok_or(CmdError::BadReply)?
            != 0)
    }

    pub fn set_rain_fog(&self, on: bool) -> Result<(), CmdError> {
        self.cmd(cmd_set::LIDAR, lidar::RAIN_FOG, &[on as u8])
            .map(drop)
    }

    /// IMU push rate: false off, true 200 Hz (Horizon, Tele-15, Avia).
    pub fn set_imu(&self, on: bool) -> Result<(), CmdError> {
        self.cmd(cmd_set::LIDAR, lidar::SET_IMU_RATE, &[on as u8])
            .map(drop)
    }

    pub fn extrinsic(&self) -> Result<Extrinsic, CmdError> {
        let p = self.cmd(cmd_set::LIDAR, lidar::GET_EXTRINSIC, &[])?;
        Extrinsic::parse(&p).ok_or(CmdError::BadReply)
    }

    pub fn set_extrinsic(&self, e: &Extrinsic) -> Result<(), CmdError> {
        self.cmd(cmd_set::LIDAR, lidar::SET_EXTRINSIC, &e.to_bytes())
            .map(drop)
    }

    /// Writes every data-port datagram from now on to an LVXR2 file (see [`crate::record`]).
    pub fn start_recording(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let writer = record::Writer::create(path, self.inner.cfg.dev_type, self.stream_profile())?;
        *lock(&self.inner.recorder) = Some(writer);
        Ok(())
    }

    /// Returns packets written, if a recording was running.
    pub fn stop_recording(&self) -> io::Result<Option<u64>> {
        match lock(&self.inner.recorder).take() {
            Some(w) => w.finish().map(Some),
            None => Ok(None),
        }
    }

    pub fn disconnect(&self) {
        if !self.inner.running.load(Ordering::Relaxed) {
            return;
        }
        if self.inner.connected.load(Ordering::Relaxed) {
            let _ = self.inner.request_typed(
                ptype::CMD,
                cmd_set::GENERAL,
                general::DISCONNECT,
                &[],
                Duration::from_millis(300),
            );
        }
        let _ = self.stop_recording();
        self.inner.running.store(false, Ordering::Relaxed);
        self.inner.connected.store(false, Ordering::Relaxed);
        self.inner.ack_cv.notify_all();
        for t in lock(&self.threads).drain(..) {
            let _ = t.join();
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        self.disconnect();
    }
}

// ---------------------------------------------------------------------------------------------
// Replay
// ---------------------------------------------------------------------------------------------

/// Plays an LVXR recording (either version) through the same sink a live device uses, at
/// recorded pace.
pub struct Replay {
    sink: Arc<Sink>,
    running: Arc<AtomicBool>,
    loops: Arc<AtomicU32>,
    thread: Option<JoinHandle<()>>,
    pub dev_type: DeviceType,
    pub stream_profile: StreamProfile,
}

impl Replay {
    pub fn open(path: impl AsRef<Path>, speed: f64, looped: bool) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let reader = record::Reader::open(&path)?;
        let dev_type = reader.dev_type;
        let stream_profile = reader.stream_profile;
        let sink = Arc::new(Sink::new(dev_type, 0, DEFAULT_MAX_BUFFERED_POINTS));
        sink.set_profile(stream_profile);
        let running = Arc::new(AtomicBool::new(true));
        let loops = Arc::new(AtomicU32::new(0));
        let (sk, run, lp) = (sink.clone(), running.clone(), loops.clone());
        let speed = speed.max(0.01);
        let thread = thread::Builder::new()
            .name("livox-replay".into())
            .spawn(move || {
                let mut reader = reader;
                loop {
                    let start = Instant::now();
                    while run.load(Ordering::Relaxed) {
                        let Ok(Some((ns, pkt))) = reader.next_packet() else {
                            break;
                        };
                        let due = start + Duration::from_nanos((ns as f64 / speed) as u64);
                        let now = Instant::now();
                        if due > now {
                            thread::sleep(due - now);
                        }
                        sk.ingest(&pkt);
                    }
                    if !looped || !run.load(Ordering::Relaxed) {
                        break;
                    }
                    match record::Reader::open(&path) {
                        Ok(r) => reader = r,
                        Err(_) => break,
                    }
                    sk.reset_clock();
                    lp.fetch_add(1, Ordering::Relaxed);
                }
            })?;
        Ok(Self {
            sink,
            running,
            loops,
            thread: Some(thread),
            dev_type,
            stream_profile,
        })
    }

    pub fn drain(&self) -> PointBatch {
        self.sink.drain()
    }

    pub fn stats(&self) -> Stats {
        self.sink.stats()
    }

    pub fn imu(&self) -> Option<Imu> {
        self.sink.imu()
    }

    pub fn loops(&self) -> u32 {
        self.loops.load(Ordering::Relaxed)
    }

    pub fn finished(&self) -> bool {
        self.thread.as_ref().is_none_or(|t| t.is_finished())
    }
}

impl Drop for Replay {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

#[cfg(test)]
mod stream_tests {
    use super::*;

    fn dual_packet(ts: u64) -> Vec<u8> {
        let mut b = vec![5, 0, 1, 0, 0, 0, 0, 0, 0, 0];
        b.extend_from_slice(&ts.to_le_bytes());
        for _ in 0..point::MID40_RECORDS_PER_PACKET {
            b.extend_from_slice(&1000i32.to_le_bytes());
            b.extend_from_slice(&[0; 8]);
            b.push(200);
        }
        b
    }

    #[test]
    fn dual_timing_and_loss_account_for_firings_not_echoes() {
        let sink = Sink::new(DeviceType::Mid40, 0, 1000);
        sink.set_profile(StreamProfile::Mid40Dual);
        sink.ingest(&dual_packet(0));
        sink.ingest(&dual_packet(500_000));
        sink.ingest(&dual_packet(1_500_000)); // one missing packet
        let b = sink.drain();
        assert_eq!(b.len(), 300);
        assert_eq!(b.t[0], b.t[1]);
        assert!((b.t[2] - 0.00001).abs() < 1e-8);
        assert!((b.t[99] - 0.00049).abs() < 1e-8);
        assert!((b.t[100] - 0.0005).abs() < 1e-8);
        assert_eq!(sink.stats().lost_packets, 1);
        assert_eq!((b.attr[1] >> 16) & 255, 1);
    }

    #[test]
    fn dual_ingest_respects_buffer_limit_mid_packet() {
        let sink = Sink::new(DeviceType::Mid40, 0, 3);
        sink.set_profile(StreamProfile::Mid40Dual);
        sink.ingest(&dual_packet(0));
        assert_eq!(sink.drain().len(), 3);
        assert_eq!(sink.stats().dropped_points, 97);
    }

    #[test]
    fn clock_jump_shifts_base_instead_of_counting_loss() {
        let sink = Sink::new(DeviceType::Mid40, 0, 1000);
        sink.set_profile(StreamProfile::Mid40Dual);
        sink.ingest(&dual_packet(0));
        sink.ingest(&dual_packet(500_000 + CLOCK_JUMP_NS));
        let b = sink.drain();
        assert_eq!(sink.stats().lost_packets, 0);
        assert!(
            (b.t[100] - 0.0005).abs() < 1e-8,
            "t stays continuous across the jump"
        );
    }
}
