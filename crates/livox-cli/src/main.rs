//! `livox`: discover, inspect, re-address and stream Livox SDK1 LiDARs from the terminal.

use clap::{Args, Parser, Subcommand, ValueEnum};
use livox::{
    host_for, local_ipv4, ConnectConfig, Device, DeviceType, Discovery, Heartbeat, IpInfo,
    StatusBits,
};
use std::error::Error;
use std::fmt;
use std::net::Ipv4Addr;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::thread;
use std::time::{Duration, Instant};

#[derive(Parser)]
#[command(
    name = "livox",
    about = "Control Livox SDK1 LiDARs (Mid-40/70, Horizon, Tele-15, Avia)"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Args, Clone)]
struct Target {
    /// LiDAR IP address
    #[arg(long)]
    lidar: Ipv4Addr,
    /// Local IP the LiDAR should talk to (default: the local address on the LiDAR's subnet)
    #[arg(long)]
    host: Option<Ipv4Addr>,
    /// Device type, if discovery cannot hear the LiDAR
    #[arg(long = "type", value_enum)]
    dev_type: Option<DeviceKind>,
}

/// The SDK1 models, as `--type` accepts them.
#[derive(Clone, Copy, ValueEnum)]
enum DeviceKind {
    Mid40,
    Mid70,
    Horizon,
    Tele15,
    Avia,
}

impl From<DeviceKind> for DeviceType {
    fn from(kind: DeviceKind) -> Self {
        match kind {
            DeviceKind::Mid40 => DeviceType::Mid40,
            DeviceKind::Mid70 => DeviceType::Mid70,
            DeviceKind::Horizon => DeviceType::Horizon,
            DeviceKind::Tele15 => DeviceType::Tele15,
            DeviceKind::Avia => DeviceType::Avia,
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
enum WorkMode {
    Normal,
    PowerSave,
    Standby,
}

impl WorkMode {
    /// The protocol's LidarMode code (see [`Device::set_mode`]).
    fn code(self) -> u8 {
        match self {
            WorkMode::Normal => 1,
            WorkMode::PowerSave => 2,
            WorkMode::Standby => 3,
        }
    }
}

impl fmt::Display for WorkMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The spelling clap accepts on the command line.
        f.write_str(
            self.to_possible_value()
                .expect("no skipped variants")
                .get_name(),
        )
    }
}

#[derive(Subcommand)]
enum Cmd {
    /// List this machine's IPv4 addresses
    Ifaces,
    /// Listen for LiDAR broadcasts
    Discover {
        #[arg(long, default_value_t = 3.0)]
        secs: f64,
    },
    /// Connect and print firmware, IP config, health, return mode and extrinsics
    Info(Target),
    /// Change the LiDAR's IP configuration (applies after a reboot)
    SetIp {
        #[command(flatten)]
        target: Target,
        /// New static address as a.b.c.d/prefix (prefix defaults to 24)
        #[arg(long = "static", conflicts_with = "dynamic")]
        static_ip: Option<String>,
        /// Gateway for the static address (default: .1 of its subnet)
        #[arg(long)]
        gateway: Option<Ipv4Addr>,
        /// Switch to DHCP instead
        #[arg(long)]
        dynamic: bool,
        /// Reboot afterwards so the change takes effect
        #[arg(long)]
        reboot: bool,
    },
    /// Reboot the LiDAR
    Reboot(Target),
    /// Set the work mode
    Mode {
        #[command(flatten)]
        target: Target,
        #[arg(value_enum)]
        mode: WorkMode,
    },
    /// Start sampling and print throughput; optionally record raw packets for replay
    Stream {
        #[command(flatten)]
        target: Target,
        #[arg(long, default_value_t = 10.0)]
        secs: f64,
        /// Write an LVXR2 recording (replayable with `livox-warp --replay`)
        #[arg(long, value_name = "FILE")]
        record: Option<PathBuf>,
    },
}

fn main() -> ExitCode {
    match run(Cli::parse().cmd) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

type Res<T> = Result<T, Box<dyn Error>>;

fn run(cmd: Cmd) -> Res<()> {
    match cmd {
        Cmd::Ifaces => {
            for i in local_ipv4() {
                println!("{:<40} {:<16} mask {}", i.name, i.ip, i.mask);
            }
            Ok(())
        }
        Cmd::Discover { secs } => discover(secs),
        Cmd::Info(t) => info(&connect(&t)?),
        Cmd::SetIp {
            target,
            static_ip,
            gateway,
            dynamic,
            reboot,
        } => {
            let new = if dynamic {
                IpInfo {
                    dynamic: true,
                    ip: Ipv4Addr::UNSPECIFIED,
                    mask: Ipv4Addr::UNSPECIFIED,
                    gateway: Ipv4Addr::UNSPECIFIED,
                }
            } else {
                let spec = static_ip.ok_or("pass --static a.b.c.d/prefix or --dynamic")?;
                let (ip, mask) = parse_cidr(&spec)?;
                let gw = gateway
                    .unwrap_or_else(|| Ipv4Addr::from((u32::from(ip) & u32::from(mask)) | 1));
                IpInfo {
                    dynamic: false,
                    ip,
                    mask,
                    gateway: gw,
                }
            };
            let dev = connect(&target)?;
            if let Ok(cur) = dev.ip_info() {
                println!("current : {cur}");
            }
            dev.set_ip(&new)?;
            println!("written : {new}");
            if reboot {
                dev.reboot(100)?;
                println!("rebooting; the LiDAR comes back in ~15 s at its new address");
            } else {
                println!(
                    "applies after the next reboot (livox reboot --lidar {})",
                    target.lidar
                );
            }
            Ok(())
        }
        Cmd::Reboot(t) => {
            connect(&t)?.reboot(100)?;
            println!("rebooting");
            Ok(())
        }
        Cmd::Mode { target, mode } => {
            connect(&target)?.set_mode(mode.code())?;
            println!("mode set to {mode}");
            Ok(())
        }
        Cmd::Stream {
            target,
            secs,
            record,
        } => stream(&connect(&target)?, secs, record.as_deref()),
    }
}

fn parse_cidr(s: &str) -> Res<(Ipv4Addr, Ipv4Addr)> {
    let (ip, prefix) = s.split_once('/').unwrap_or((s, "24"));
    let ip: Ipv4Addr = ip.parse().map_err(|_| format!("bad address {ip:?}"))?;
    let prefix: u32 = prefix
        .parse()
        .map_err(|_| format!("bad prefix {prefix:?}"))?;
    if prefix == 0 || prefix > 30 {
        return Err("prefix must be 1..=30".into());
    }
    Ok((ip, Ipv4Addr::from(u32::MAX << (32 - prefix))))
}

fn discover(secs: f64) -> Res<()> {
    let disc = Discovery::start().map_err(|e| format!("cannot listen on UDP 55000: {e}"))?;
    thread::sleep(Duration::from_secs_f64(secs));
    let found = disc.devices(Duration::from_secs_f64(secs + 1.0));
    if found.is_empty() {
        if disc.datagrams() == 0 {
            println!("nothing heard on UDP 55000 in {secs:.0} s.");
            println!("If Livox Viewer is open, close it: it binds the port to one address and hides broadcasts.");
        } else {
            println!(
                "{} datagrams on 55000 but no valid broadcasts ({} bad frames)",
                disc.datagrams(),
                disc.bad_frames()
            );
        }
        return Ok(());
    }
    println!(
        "{:<18} {:<9} {:<16} host address on its subnet",
        "broadcast code", "type", "lidar ip"
    );
    for s in found {
        let host = match host_for(s.ip) {
            Some(i) => format!("{} ({})", i.ip, i.name),
            None => format!("NONE - add one, e.g. {}", suggest_host(s.ip)),
        };
        println!("{:<18} {:<9} {:<16} {host}", s.code, s.dev_type, s.ip);
    }
    Ok(())
}

/// A free-looking address on the LiDAR's /24 for the host to take.
fn suggest_host(lidar: Ipv4Addr) -> Ipv4Addr {
    let o = lidar.octets();
    Ipv4Addr::new(o[0], o[1], o[2], if o[3] == 50 { 51 } else { 50 })
}

/// Why a connection cannot start, with the command that adds a suitable address on this OS.
fn no_host_error(lidar: Ipv4Addr) -> String {
    let host = suggest_host(lidar);
    let hint = if cfg!(windows) {
        format!(
            "netsh interface ipv4 set interface Ethernet dhcpstaticipcoexistence=enabled\n  \
             netsh interface ipv4 add address Ethernet {host} 255.255.255.0"
        )
    } else {
        format!("sudo ip addr add {host}/24 dev eth0")
    };
    format!("no local address on {lidar}'s subnet. Add one (adjust the adapter name), e.g.\n  {hint}\nor pass --host.")
}

fn connect(t: &Target) -> Res<Device> {
    let host = match t.host {
        Some(h) => h,
        None => host_for(t.lidar)
            .map(|i| i.ip)
            .ok_or_else(|| no_host_error(t.lidar))?,
    };
    let dev_type = match t.dev_type {
        Some(kind) => kind.into(),
        None => discover_type(t.lidar).unwrap_or(DeviceType::Unknown(255)),
    };
    let dev = Device::connect(ConnectConfig::new(t.lidar, host, dev_type)).map_err(|e| {
        format!(
            "handshake with {} via {host} failed: {e}. Is another program connected to it?",
            t.lidar
        )
    })?;
    let fw = dev
        .status()
        .firmware
        .map(|f| format!(", firmware {:02}.{:02}.{:02}{:02}", f[0], f[1], f[2], f[3]));
    println!(
        "connected: {dev_type} at {} via {host}{}",
        t.lidar,
        fw.unwrap_or_default()
    );
    Ok(dev)
}

fn discover_type(lidar: Ipv4Addr) -> Option<DeviceType> {
    let disc = Discovery::start().ok()?;
    let until = Instant::now() + Duration::from_millis(2500);
    while Instant::now() < until {
        if let Some(s) = disc
            .devices(Duration::from_secs(5))
            .into_iter()
            .find(|s| s.ip == lidar)
        {
            return Some(s.dev_type);
        }
        thread::sleep(Duration::from_millis(100));
    }
    None
}

/// The first heartbeat reply, which normally lands within a second of connecting.
fn wait_for_heartbeat(dev: &Device, cap: Duration) -> Option<Heartbeat> {
    let deadline = Instant::now() + cap;
    loop {
        let hb = dev.status().heartbeat;
        if hb.is_some() || Instant::now() >= deadline {
            return hb;
        }
        thread::sleep(Duration::from_millis(50));
    }
}

fn info(dev: &Device) -> Res<()> {
    match wait_for_heartbeat(dev, Duration::from_millis(1500)) {
        Some(hb) => {
            println!("state   : {}", hb.state_name());
            let problems = StatusBits(hb.status).problems();
            println!(
                "health  : {}",
                if problems.is_empty() {
                    "ok".to_string()
                } else {
                    problems.join(", ")
                }
            );
        }
        None => println!("state   : (no heartbeat yet)"),
    }
    match dev.ip_info() {
        Ok(ip) => println!("ip      : {ip}"),
        Err(e) => println!("ip      : ({e})"),
    }
    match dev.return_mode() {
        Ok(m) => println!(
            "returns : {}",
            ["single first", "single strongest", "dual", "triple"]
                .get(m as usize)
                .unwrap_or(&"?")
        ),
        Err(e) => println!("returns : ({e})"),
    }
    match dev.extrinsic() {
        Ok(e) => println!(
            "mount   : roll {:.2} pitch {:.2} yaw {:.2} deg, xyz {} {} {} mm",
            e.roll, e.pitch, e.yaw, e.x, e.y, e.z
        ),
        Err(e) => println!("mount   : ({e})"),
    }
    Ok(())
}

fn stream(dev: &Device, secs: f64, record: Option<&Path>) -> Res<()> {
    if let Some(path) = record {
        dev.start_recording(path)
            .map_err(|e| format!("cannot record to {}: {e}", path.display()))?;
    }
    dev.start_sampling()?;
    let start = Instant::now();
    let mut total = 0usize;
    while start.elapsed().as_secs_f64() < secs {
        thread::sleep(Duration::from_secs(1));
        let batch = dev.drain();
        total += batch.len();
        let s = dev.stats();
        let st = dev.status();
        println!(
            "{:5.1}s  {:>7} pts/s  packets {:>7}  lost {:>4}  data_type {}  {}",
            start.elapsed().as_secs_f64(),
            batch.len(),
            s.packets,
            s.lost_packets,
            s.data_type,
            if st.connected { "" } else { "HEARTBEAT LOST" }
        );
    }
    dev.stop_sampling()?;
    if let (Ok(Some(n)), Some(path)) = (dev.stop_recording(), record) {
        println!("recorded {n} packets to {}", path.display());
    }
    println!("{total} points in {:.1} s", start.elapsed().as_secs_f64());
    Ok(())
}
