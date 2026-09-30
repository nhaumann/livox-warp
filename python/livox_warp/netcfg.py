"""Host network configuration for reaching a Livox sensor.

A Livox unit ships with a static address on its own /24 subnet and streams to whatever host
address it is given at connection time, so the host needs an address on that subnet. Livox's own
tools expect the host at .50 of the sensor's subnet; suggest_host() follows that convention and
add_host_alias() gives an adapter such an address without giving up the one DHCP assigned.
"""

from __future__ import annotations

import ipaddress
import os
import subprocess
import sys
import tempfile


def suggest_host(lidar_ip: str) -> str:
    """The host address Livox tools expect on a sensor's subnet: .50, or .51 when the sensor itself is .50.

    Raises ValueError on a malformed address.
    """
    lidar = ipaddress.IPv4Address(lidar_ip)
    subnet = ipaddress.IPv4Network((lidar, 24), strict=False)
    return str(subnet.network_address + (51 if int(lidar) & 0xFF == 50 else 50))


def add_host_alias(adapter: str, ip: str, mask: str = "255.255.255.0") -> None:
    """Give `adapter` a second IPv4 address (keeping DHCP) so a sensor on another subnet is reachable.

    Windows only: runs netsh in an elevated command prompt, so it raises a UAC prompt and waits for
    it. Raises NotImplementedError on other platforms, ValueError on a malformed address or mask, and
    subprocess.CalledProcessError when the elevated prompt is refused. netsh's own result is not
    reported back (the prompt is hidden): re-read the adapter's addresses afterwards.
    """
    if sys.platform != "win32":
        raise NotImplementedError("add_host_alias configures Windows adapters with netsh; add the address by hand")
    if '"' in adapter:
        raise ValueError(f"adapter name cannot contain a double quote: {adapter!r}")
    ip, mask = str(ipaddress.IPv4Address(ip)), str(ipaddress.IPv4Address(mask))
    script = (
        "@echo off\r\n"
        f'netsh interface ipv4 set interface "{adapter}" dhcpstaticipcoexistence=enabled\r\n'
        f'netsh interface ipv4 add address "{adapter}" {ip} {mask}\r\n'
    )
    fd, path = tempfile.mkstemp(suffix=".cmd")
    with os.fdopen(fd, "w", newline="") as f:  # newline="": keep the "\r\n" line ends cmd.exe expects
        f.write(script)
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"Start-Process cmd.exe -ArgumentList '/c \"{path}\"' -Verb RunAs -Wait -WindowStyle Hidden"],
            check=True,
        )
    finally:
        os.unlink(path)
