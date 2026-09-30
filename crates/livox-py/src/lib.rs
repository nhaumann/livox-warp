//! `livox_warp._native`: the Rust LiDAR stack exposed to Python.
//!
//! Points cross the boundary as three `bytes` objects (xyz f32x3, attr u32, t f32), each copied
//! once out of the Rust buffers; numpy then views them with `frombuffer` without another copy.
//! Every blocking network call releases the GIL.

use livox::{ConnectConfig, DeviceType, Extrinsic, IpInfo, PointBatch, Stats, StatusBits};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use std::net::Ipv4Addr;
use std::path::PathBuf;
use std::time::Duration;

fn err<E: std::fmt::Display>(e: E) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

fn ip(s: &str) -> PyResult<Ipv4Addr> {
    s.parse()
        .map_err(|_| PyValueError::new_err(format!("bad IPv4 address {s:?}")))
}

/// Element types whose storage can be handed to Python byte for byte: no padding, and every
/// bit pattern a valid value. Private to this crate, so nothing else can claim to be one.
trait Pod: Copy {}
impl Pod for f32 {}
impl Pod for u32 {}

fn as_bytes<T: Pod>(v: &[T]) -> &[u8] {
    // SAFETY: `Pod` guarantees every byte of every element is initialised, and the length is
    // the slice's size in bytes, so the view covers exactly its elements and nothing else.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

type Drained<'py> = (
    Bound<'py, PyBytes>,
    Bound<'py, PyBytes>,
    Bound<'py, PyBytes>,
    usize,
);

fn batch_to_py(py: Python<'_>, b: PointBatch) -> Drained<'_> {
    (
        PyBytes::new(py, as_bytes(&b.xyz)),
        PyBytes::new(py, as_bytes(&b.attr)),
        PyBytes::new(py, as_bytes(&b.t)),
        b.len(),
    )
}

fn stats_to_py(py: Python<'_>, s: Stats) -> PyResult<Bound<'_, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("packets", s.packets)?;
    d.set_item("points", s.points)?;
    d.set_item("bytes", s.bytes)?;
    d.set_item("lost_packets", s.lost_packets)?;
    d.set_item("bad_packets", s.bad_packets)?;
    d.set_item("imu_samples", s.imu_samples)?;
    d.set_item("dropped_points", s.dropped_points)?;
    d.set_item("data_type", s.data_type)?;
    d.set_item("timestamp_type", s.timestamp_type)?;
    d.set_item("data_status", s.data_status)?;
    d.set_item("problems", StatusBits(s.data_status).problems())?;
    Ok(d)
}

type ImuTuple = ((f32, f32, f32), (f32, f32, f32));

fn imu_to_py(imu: Option<livox::Imu>) -> Option<ImuTuple> {
    imu.map(|i| {
        (
            (i.gyro[0], i.gyro[1], i.gyro[2]),
            (i.acc[0], i.acc[1], i.acc[2]),
        )
    })
}

#[pyfunction]
fn local_interfaces() -> Vec<(String, String, String)> {
    livox::local_ipv4()
        .into_iter()
        .map(|i| (i.name, i.ip.to_string(), i.mask.to_string()))
        .collect()
}

/// The local address on `lidar_ip`'s subnet, or None.
#[pyfunction]
fn host_for(lidar_ip: &str) -> PyResult<Option<String>> {
    Ok(livox::host_for(ip(lidar_ip)?).map(|i| i.ip.to_string()))
}

#[pyfunction]
fn device_type_name(dev_type: u8) -> String {
    DeviceType::from_u8(dev_type).to_string()
}

#[pyfunction]
fn status_problems(word: u32) -> Vec<String> {
    StatusBits(word).problems()
}

#[pyclass(frozen, name = "Discovery", module = "livox_warp._native")]
struct PyDiscovery {
    inner: livox::Discovery,
}

#[pymethods]
impl PyDiscovery {
    #[new]
    fn new() -> PyResult<Self> {
        Ok(Self {
            inner: livox::Discovery::start().map_err(err)?,
        })
    }

    /// LiDARs heard in the last `max_age` seconds.
    #[pyo3(signature = (max_age = 3.0))]
    fn devices<'py>(&self, py: Python<'py>, max_age: f64) -> PyResult<Vec<Bound<'py, PyDict>>> {
        self.inner
            .devices(Duration::from_secs_f64(max_age))
            .into_iter()
            .map(|s| {
                let d = PyDict::new(py);
                d.set_item("code", &s.code)?;
                d.set_item("type", s.dev_type.as_u8())?;
                d.set_item("type_name", s.dev_type.to_string())?;
                d.set_item("ip", s.ip.to_string())?;
                d.set_item("age", s.last_seen.elapsed().as_secs_f64())?;
                d.set_item("broadcasts", s.broadcasts)?;
                d.set_item("host", livox::host_for(s.ip).map(|i| i.ip.to_string()))?;
                Ok(d)
            })
            .collect()
    }

    fn datagrams(&self) -> u64 {
        self.inner.datagrams()
    }
}

#[pyclass(frozen, name = "Device", module = "livox_warp._native")]
struct PyDevice {
    inner: livox::Device,
}

#[pymethods]
impl PyDevice {
    /// Handshake with the LiDAR at `lidar_ip`, telling it to stream to `host_ip`.
    ///
    /// The keyword arguments are the ports the LiDAR sends to (`imu_port` defaults to
    /// `data_port`), the device index stamped into each point's attr word, and how many points
    /// are held between `drain()` calls before new ones are dropped.
    #[new]
    #[pyo3(signature = (
        lidar_ip,
        host_ip,
        dev_type = 255,
        *,
        cmd_port = livox::DEFAULT_CMD_PORT,
        data_port = livox::DEFAULT_DATA_PORT,
        imu_port = None,
        dev_index = 0,
        max_buffered_points = livox::DEFAULT_MAX_BUFFERED_POINTS,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        lidar_ip: &str,
        host_ip: &str,
        dev_type: u8,
        cmd_port: u16,
        data_port: u16,
        imu_port: Option<u16>,
        dev_index: u8,
        max_buffered_points: usize,
    ) -> PyResult<Self> {
        let cfg = ConnectConfig {
            cmd_port,
            data_port,
            imu_port: imu_port.unwrap_or(data_port),
            dev_index,
            max_buffered_points,
            ..ConnectConfig::new(ip(lidar_ip)?, ip(host_ip)?, DeviceType::from_u8(dev_type))
        };
        let dev = py.detach(|| livox::Device::connect(cfg)).map_err(err)?;
        Ok(Self { inner: dev })
    }

    /// (xyz_bytes, attr_bytes, t_bytes, n) for everything received since the last call.
    fn drain<'py>(&self, py: Python<'py>) -> Drained<'py> {
        batch_to_py(py, self.inner.drain())
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        stats_to_py(py, self.inner.stats())
    }

    fn status<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = self.inner.status();
        let d = PyDict::new(py);
        d.set_item("connected", s.connected)?;
        d.set_item("state", s.heartbeat.map(|h| h.state))?;
        d.set_item("state_name", s.heartbeat.map(|h| h.state_name()))?;
        d.set_item("status_word", s.heartbeat.map(|h| h.status))?;
        d.set_item(
            "problems",
            s.heartbeat
                .map(|h| StatusBits(h.status).problems())
                .unwrap_or_default(),
        )?;
        d.set_item("heartbeat_age", s.heartbeat_age.map(|a| a.as_secs_f64()))?;
        d.set_item("missed_heartbeats", s.missed_heartbeats)?;
        d.set_item(
            "firmware",
            s.firmware
                .map(|f| format!("{:02}.{:02}.{:02}{:02}", f[0], f[1], f[2], f[3])),
        )?;
        d.set_item("stream_profile", self.inner.stream_profile().name())?;
        d.set_item("abnormal", s.abnormal)?;
        d.set_item("data_datagrams", self.inner.data_datagrams())?;
        let cfg = self.inner.config();
        d.set_item("lidar_ip", cfg.lidar_ip.to_string())?;
        d.set_item("host_ip", cfg.host_ip.to_string())?;
        d.set_item("type", cfg.dev_type.as_u8())?;
        d.set_item("type_name", cfg.dev_type.to_string())?;
        Ok(d)
    }

    fn imu(&self) -> Option<ImuTuple> {
        imu_to_py(self.inner.imu())
    }

    fn start_sampling(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.start_sampling()).map_err(err)
    }

    fn stop_sampling(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.stop_sampling()).map_err(err)
    }

    fn ip_info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let info = py.detach(|| self.inner.ip_info()).map_err(err)?;
        let d = PyDict::new(py);
        d.set_item("dynamic", info.dynamic)?;
        d.set_item("ip", info.ip.to_string())?;
        d.set_item("mask", info.mask.to_string())?;
        d.set_item("gateway", info.gateway.to_string())?;
        Ok(d)
    }

    /// Static when `dynamic` is False. Applies after `reboot()`.
    #[pyo3(signature = (dynamic, ip_addr = "0.0.0.0", mask = "255.255.255.0", gateway = "0.0.0.0"))]
    fn set_ip(
        &self,
        py: Python<'_>,
        dynamic: bool,
        ip_addr: &str,
        mask: &str,
        gateway: &str,
    ) -> PyResult<()> {
        let info = IpInfo {
            dynamic,
            ip: ip(ip_addr)?,
            mask: ip(mask)?,
            gateway: ip(gateway)?,
        };
        py.detach(|| self.inner.set_ip(&info)).map_err(err)
    }

    #[pyo3(signature = (delay_ms = 100))]
    fn reboot(&self, py: Python<'_>, delay_ms: u16) -> PyResult<()> {
        py.detach(|| self.inner.reboot(delay_ms)).map_err(err)
    }

    /// 1 normal, 2 power-saving, 3 standby.
    fn set_mode(&self, py: Python<'_>, mode: u8) -> PyResult<()> {
        py.detach(|| self.inner.set_mode(mode)).map_err(err)
    }

    /// 0 single first, 1 single strongest, 2 dual, 3 triple.
    fn set_return_mode(&self, py: Python<'_>, mode: u8) -> PyResult<()> {
        py.detach(|| self.inner.set_return_mode(mode)).map_err(err)
    }

    fn return_mode(&self, py: Python<'_>) -> PyResult<u8> {
        py.detach(|| self.inner.return_mode()).map_err(err)
    }

    fn set_fan(&self, py: Python<'_>, on: bool) -> PyResult<()> {
        py.detach(|| self.inner.set_fan(on)).map_err(err)
    }

    fn set_rain_fog(&self, py: Python<'_>, on: bool) -> PyResult<()> {
        py.detach(|| self.inner.set_rain_fog(on)).map_err(err)
    }

    fn set_imu(&self, py: Python<'_>, on: bool) -> PyResult<()> {
        py.detach(|| self.inner.set_imu(on)).map_err(err)
    }

    /// (roll, pitch, yaw) in degrees and (x, y, z) in millimetres, as stored on the LiDAR.
    fn extrinsic(&self, py: Python<'_>) -> PyResult<(f32, f32, f32, i32, i32, i32)> {
        let e = py.detach(|| self.inner.extrinsic()).map_err(err)?;
        Ok((e.roll, e.pitch, e.yaw, e.x, e.y, e.z))
    }

    #[allow(clippy::too_many_arguments)]
    fn set_extrinsic(
        &self,
        py: Python<'_>,
        roll: f32,
        pitch: f32,
        yaw: f32,
        x: i32,
        y: i32,
        z: i32,
    ) -> PyResult<()> {
        let e = Extrinsic {
            roll,
            pitch,
            yaw,
            x,
            y,
            z,
        };
        py.detach(|| self.inner.set_extrinsic(&e)).map_err(err)
    }

    /// Record every data-port datagram to an LVXR2 file at `path` (str or os.PathLike).
    fn start_recording(&self, path: PathBuf) -> PyResult<()> {
        self.inner.start_recording(path).map_err(err)
    }

    /// Packets written, or None if nothing was recording.
    fn stop_recording(&self) -> PyResult<Option<u64>> {
        self.inner.stop_recording().map_err(err)
    }

    fn disconnect(&self, py: Python<'_>) {
        py.detach(|| self.inner.disconnect())
    }
}

#[pyclass(frozen, name = "Replay", module = "livox_warp._native")]
struct PyReplay {
    inner: livox::Replay,
}

#[pymethods]
impl PyReplay {
    /// Play the LVXR recording at `path` (str or os.PathLike) at `speed` times real time.
    #[new]
    #[pyo3(signature = (path, speed = 1.0, looped = true))]
    fn new(path: PathBuf, speed: f64, looped: bool) -> PyResult<Self> {
        Ok(Self {
            inner: livox::Replay::open(path, speed, looped).map_err(err)?,
        })
    }

    fn drain<'py>(&self, py: Python<'py>) -> Drained<'py> {
        batch_to_py(py, self.inner.drain())
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        stats_to_py(py, self.inner.stats())
    }

    fn imu(&self) -> Option<ImuTuple> {
        imu_to_py(self.inner.imu())
    }

    fn loops(&self) -> u32 {
        self.inner.loops()
    }

    fn finished(&self) -> bool {
        self.inner.finished()
    }

    #[getter]
    fn dev_type(&self) -> u8 {
        self.inner.dev_type.as_u8()
    }

    #[getter]
    fn stream_profile(&self) -> &'static str {
        self.inner.stream_profile.name()
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(local_interfaces, m)?)?;
    m.add_function(wrap_pyfunction!(host_for, m)?)?;
    m.add_function(wrap_pyfunction!(device_type_name, m)?)?;
    m.add_function(wrap_pyfunction!(status_problems, m)?)?;
    m.add_class::<PyDiscovery>()?;
    m.add_class::<PyDevice>()?;
    m.add_class::<PyReplay>()?;
    Ok(())
}
