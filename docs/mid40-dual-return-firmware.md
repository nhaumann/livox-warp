# Mid-40 dual-return firmware

The Mid-40 ships with single-return firmware. Livox publishes two special firmware images for the
Mid series, and this project supports both:

| image | version | behaviour | MD5 |
|---|---|---|---|
| `LIVOX_MID_FW_03.03.0006.bin` | 03.03.0006 | dual return: a second echo per firing, reported with a fixed reflectivity of 200 | `9FA06C83017A42E2A05EBB92CD51FAD8` |
| `LIVOX_MID_FW_03.03.0004.bin` | 03.03.0004 | single return with strict threadlike-noise filtering (the usual factory image) | `7F1BB25C32C5266060F8DC50D5C8E3D2` |
| `LIVOX_MID_FW_03.08.0000.bin` | 03.08.0000 | vendor standard firmware | `56AD978F521828688C6BD7574C97BF04` (locally computed, not vendor-verified) |

Sources:

- https://github.com/Livox-SDK/Special-Firmwares-for-Livox-LiDARs/tree/master/Multi-return_Firmware_For_Livox_MID
- https://github.com/Livox-SDK/Special-Firmwares-for-Livox-LiDARs/tree/master/Threadlike-Noise_Filtering_Firmware_For_Livox_MID
- https://www.livoxtech.com/downloads

The images are Livox's and are not part of this repository. The vendor README's table names the
dual-return file `0001`, but the version and MD5 it lists match the `03.03.0006` image at the path
above.

## Installing

These are complete replacement images, not additive features. Keep power and Ethernet connected
for the whole update.

1. Use Livox Viewer 0.11.0 with only the intended device connected.
2. Tools > Firmware Upgrade, choose the image in the Mid-40/100 field, select the device.
3. To return to the previous behaviour, install `03.03.0004` through the same updater.

A saved vendor image is not an on-device flash dump and does not guarantee recovery from a failed
bootloader or an interrupted flash.

## What changes for this software

With `03.03.0006` the device reports firmware `03.03.0006` and the stream profile becomes
`mid40-dual`: each firing carries two returns with a shared timestamp, and the second return's
reflectivity is the fixed marker 200 rather than a measurement. Rain/fog suppression is unavailable
on this image. In one verification run, 10,038 packets held 481,706 valid points, 466,633 first
echoes and 15,073 second echoes, with zero lost, bad or dropped packets, and the stored mounting
pose and IP configuration were unchanged by the update. The viewer's Return colour mode shows the
two echoes; the return filter (View > Denoise and filters) and the odometry's `ret_mask` select them.
Second returns are about five times noisier than first returns and sit a few centimetres in front of
the surface (measured against a terrestrial scan, `tools/calib_prior_map.py`), so the odometry keeps
both by default but a standing scanner registers with less jitter on first returns only.

The earlier host code read the version bytes in reverse and displayed `04.00.0303`; the Rust crate
decodes them in the documented order.
