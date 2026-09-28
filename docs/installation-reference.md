# Installation reference

Normal setup: [Jetson guide](jetson.md), [Mac app](macos-app.md),
[PC guide](platforms.md). This page: manual installs and custom integrations.

## Jetson installation

### What the installer changes

- Installs Docker and NVIDIA's container toolkit if missing.
- Downloads the server image, or builds it on the Jetson when none exists for
  its JetPack (10 to 30 minutes).
- Checks the server can use the GPU.
- Sets the MAXN SUPER power mode, which the large models need (the supply must
  deliver it; may need one restart, which the installer reports).
- Adds 8 GB of swap for preparing the 1.7 GB models.
- Installs the `jetlink-server` boot service and the `jetlink` command.
- Stops boot waiting for a network (none in the car; cost about two minutes)
  and caps the system log at 200 MB.
- Keeps models and prepared engines in `/mnt/data/jetlink`.

### Installing by hand

The installer is the supported path. Its pieces, from a checkout:

1. Docker, and the NVIDIA Container Toolkit with `sudo nvidia-ctk runtime
   configure --runtime=docker`.
   - JetPack 6: Ubuntu's `docker.io`; Docker 28+ cannot run containers on its
     kernel.
   - Jetson: `nvidia-container-toolkit`, not JetPack's `nvidia-container`,
     which replaces Docker with the newest Docker CE in the background a minute
     after apt finishes.
2. Server image: `sudo docker/build.sh` picks `docker/Dockerfile` (CUDA 13:
   JetPack 7.2 and PCs) or `docker/Dockerfile.jetpack6`.
3. `/etc/jetlink/server.env`, read by `scripts/jetlink-run-server` to start the
   container. Its header lists every setting. `JETLINK_IMAGE` is the image ID
   from `sudo docker image inspect --format '{{.Id}}' jetlink:latest`.
4. `scripts/jetlink-run-server` as `/usr/local/lib/jetlink/run-server`,
   `scripts/jetlink-server.service` in `/etc/systemd/system`, then
   `sudo systemctl enable --now jetlink-server`.
5. Always-on supply only (lets the comma wake the Jetson):
   `scripts/99-jetlink-usb-wakeup.rules` in `/etc/udev/rules.d`,
   `scripts/jetlink-wake-setup.sh` as `/usr/local/lib/jetlink/wake-setup`;
   optionally the `scripts/jetlink-poweroff.*` units.

Foreground run from a checkout (Ctrl-C stops):

```bash
sudo docker/run.sh --transport usb
```

## Custom USB integrations

- comma 3X (AGNOS kernel 4.9.103) has FunctionFS and USB gadget support.
- The server is always the USB host: libusb, no gadget kernel modules.
- Nothing on the comma runs by hand. The owner builds the gadget on its first
  step, USB or iOS per the comma's Accelerator Link setting, and rebuilds it
  when the setting changes.

`scripts/comma/jetlink-root.sh` is every root action Jetlink takes on the comma
(comma four and 3X); the owner runs it under `sudo -n`.

| Subcommand | Does |
| --- | --- |
| `gadget` | Creates the gadget configuration: the vendor interface alone. The owner then opens `ep0`, writes the FunctionFS descriptors and binds the USB device controller (binding needs the descriptors first). |
| `gadget --ios` | Accelerator Link iOS: composite, adds a network interface for an iPhone. |
| `net` | Run by the owner after each iOS bind (the interface exists only from the first bind). |
| `port hold`, `port off` | Keeps the USB-C port the device end of a USB link. |
| `vm apply`, `vm restore` | Sets and undoes the VM tuning the link needs while the comma records. |
| `check` | By hand, `sudo scripts/comma/jetlink-root.sh check`: what is built and the negotiated bus speed. |
| `teardown` | Removes the gadget. |

The owner holds the gadget while the link is on; openpilot lists it as
`jetlinkd`. Standard library only, about 10 MB. Code in `jetlink/comma/`:

| File | Role |
| --- | --- |
| `gadget.py` | the gadget, and the openpilot params it reads by name |
| `owner.py` | the owner |
| `lending.py` | the lease modeld borrows the endpoints, or a phone's dial, on |
| `port.py` | the USB-C port |
| `root.py` | runs `jetlink-root.sh` under `sudo -n`, on AGNOS only |

The zoompilot fork starts it from a shim,
`openpilot/sunnypilot/accelerators/jetlink/owner.py`, with the fork's
provisioning worker.

| Descriptor | Value |
| --- | --- |
| idVendor:idProduct | `1209:0001` (pid.codes test allocation) |
| bcdDevice | `0x0100`; `0x0101` for iOS, so hosts refetch cached descriptors |
| bDeviceClass/SubClass/Protocol | `0x00/0x00/0x00`, class per interface; for iOS `0xEF/0x02/0x01`, Miscellaneous with interface association (composite) |
| Interface 0 | `0xFF/0xFF/0xFF` vendor specific, one bulk IN and one bulk OUT endpoint: the Jetlink link |
| Interfaces 1 and 2 (iOS only) | CDC-NCM control and data: the network for an iPhone |
| Network (iOS only) | comma `192.168.60.1/24`; DHCP `192.168.60.2` to `.254` with 10 minute leases from dnsmasq on the gadget's interface (`usb1` or later, since the modem holds `usb0`), no router or DNS options |

Custom distributions need their own USB product ID.

## Versions and manual rollback

### Choose a version

- The installer follows `main`; Mac app releases build from it.
- The zoompilot fork pins the Jetlink commit it was tested with as its
  `jetlink_repo` submodule; `main` stays compatible with the fork's current
  `jetson-trt` branch.
- On a protocol version mismatch the server rejects the connection and the
  comma drives on the small model.

To install a release or the fork's pinned commit instead of `main`, pass it to
the installer (replace `v0.4.0`):

```bash
curl -fsSL https://raw.githubusercontent.com/zoompilot/jetlink/v0.4.0/install.sh | bash -s -- --ref v0.4.0
```

### Restore a Docker image

- The image that runs is `JETLINK_IMAGE` in `/etc/jetlink/server.env` (an ID
  from `sudo docker image inspect --format '{{.Id}}' IMAGE`;
  `sudo docker image ls` lists them). Set it, then `jetlink restart`.
- Each update keeps the settings it replaced as `/etc/jetlink/server.env.prev`.
  Back one update: `sudo cp /etc/jetlink/server.env.prev /etc/jetlink/server.env`,
  then `jetlink restart`.

## Deep sleep and USB wake

**Always on** in the installer enables USB wake on the Jetson's hubs, gives the
container `/sys/power`, sets `--sleep-after 120`, and checks for `deep` in
`/sys/power/mem_sleep`.

- Ignition off: the comma releases USB once the engine is ready and at least
  one minute has passed.
- The Jetson sleeps after 120 s without a USB connection. USB connect or
  disconnect wakes it; with no new connection it sleeps again after 120 s.
- Failed sleep retries after 10 s, doubling up to 5 minutes. Check the logs,
  write access to `/sys/power`, and USB wake on the root and onboard hubs.
- Without `--sleep-after`, the link stays up while the comma is awake.

## Battery-protection shutdown

- When enabled, the comma requests shutdown at 11.8 V or after 30 hours parked.
- The server writes a flag in the models folder; `jetlink-poweroff.path`
  triggers the host service, which removes it and powers off. Flags from
  earlier boots are ignored.
- Dry run: `touch /mnt/data/jetlink/poweroff-dry-run`; remove the file to
  restore shutdown.
- Restart after full shutdown needs hardware that cycles DC power or triggers
  the J14 power-button input; the devkit boots when DC power returns.
