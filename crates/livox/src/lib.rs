//! Livox SDK1 protocol, implemented natively (no C SDK).
//!
//! Covers the LiDARs that speak the original Livox SDK protocol: Mid-40/100, Mid-70,
//! Horizon, Tele-15 and Avia. The host picks which of its own IPv4 addresses the LiDAR
//! should stream to, which is the control Livox Viewer does not give you.

pub mod crc;
pub mod net;
pub mod point;
pub mod proto;
pub mod record;

pub use net::{
    host_for, local_ipv4, CmdError, ConnectConfig, Device, Discovery, LocalIface, PointBatch,
    Replay, Seen, Sink, Stats, Status, DEFAULT_CMD_PORT, DEFAULT_DATA_PORT,
    DEFAULT_MAX_BUFFERED_POINTS,
};
pub use point::{DataHeader, Imu, Point, StreamProfile, UnknownProfile};
pub use proto::{DeviceType, Extrinsic, Heartbeat, IpInfo, StatusBits};
