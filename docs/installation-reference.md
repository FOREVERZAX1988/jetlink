# Installation reference

For normal setup, use the [Jetson guide](jetson.md), [Mac app](macos-app.md),
or [PC guide](platforms.md). This page covers manual installation and custom
integrations.

## Jetson installation

### What the installer changes

The installer:

- installs Docker and NVIDIA's container toolkit if they are missing
- downloads the Jetlink server, or builds it on the Jetson when there is no
  ready-made one for its JetPack (the same 10 to 30 minutes)
- checks that the server can use the GPU
- switches the Jetson to its fastest power mode, MAXN SUPER, which the large
  models need to keep up (the power supply has to deliver it; switching can
  need one restart, and the installer says so at the end)
- adds 8 GB of swap, which the 1.7 GB models need while they are prepared
- sets up the `jetlink-server` service to start at every boot, and the
  `jetlink` command
- stops the Jetson waiting for a network at boot (the car has none, and waiting
  cost about two minutes), and keeps the system log under 200 MB
- keeps models and prepared engines in `/mnt/data/jetlink`

### Installing by hand

The installer is the supported way. For a custom setup, these are the pieces it
puts together, from a checkout of this repository:

1. Docker, and the NVIDIA Container Toolkit with `sudo nvidia-ctk runtime
   configure --runtime=docker`. On JetPack 6 use Ubuntu's `docker.io`: Docker 28
   and later cannot run containers on a JetPack 6 kernel. On a Jetson install
   `nvidia-container-toolkit`, not JetPack's `nvidia-container`: that package
   removes whatever Docker is installed and puts in the newest Docker CE, in
   the background, a minute after apt finishes.
2. The server image: `sudo docker/build.sh` picks `docker/Dockerfile` (CUDA 13,
   JetPack 7.2 and PCs) or `docker/Dockerfile.jetpack6`.
3. `/etc/jetlink/server.env`, which `scripts/jetlink-run-server` reads to start
   the container. Its header lists every setting; `JETLINK_IMAGE` is the
   image's ID from `sudo docker image inspect --format '{{.Id}}' jetlink:latest`.
4. `scripts/jetlink-run-server` installed as `/usr/local/lib/jetlink/run-server`
   and `scripts/jetlink-server.service` in `/etc/systemd/system`, then
   `sudo systemctl enable --now jetlink-server`.
5. On an always-on supply, `scripts/99-jetlink-usb-wakeup.rules` in
   `/etc/udev/rules.d` and `scripts/jetlink-wake-setup.sh` as
   `/usr/local/lib/jetlink/wake-setup`, so the comma can wake the Jetson; and
   optionally the `scripts/jetlink-poweroff.*` units.

To try the server in a terminal first, `sudo docker/run.sh --transport usb`
runs it in the foreground; Ctrl-C stops it.
## Custom USB integrations

The comma 3X with AGNOS kernel 4.9.103 includes FunctionFS and USB gadget
support. The Jetson host uses libusb and does not need gadget kernel modules.
The server is always the USB host.

Nothing on the comma needs running by hand. The owner builds the gadget on its
first step, for USB or iOS as the comma's Accelerator Link setting says, and
rebuilds it when the setting moves.

`scripts/comma/jetlink-root.sh` is everything Jetlink does as root on the
comma, for the comma four and the comma 3X, and the owner runs it under
`sudo -n`. Its `gadget` subcommand creates the gadget configuration. The owner
then opens `ep0`, writes the FunctionFS descriptors, and binds the USB device
controller; the script cannot bind the controller before those descriptors
exist. With `gadget --ios` (Accelerator Link set to iOS) the gadget is
composite, with a network interface for an iPhone, and the owner runs `net`
after each bind, because the interface only exists from the first bind on.
Without `--ios` the gadget is the vendor interface alone. `port hold|off` keeps
the USB-C port the device end of a USB link, and `vm apply|restore` sets and
undoes the VM tuning the link needs while the comma records. By hand, `sudo
scripts/comma/jetlink-root.sh check` prints what has been built and the
negotiated bus speed, and `teardown` removes the gadget.

The owner is the process that holds the gadget for as long as the link is on;
openpilot lists it as `jetlinkd`. It and everything it uses on the comma live in
`jetlink/comma/`: the gadget and the openpilot params it reads by name
(`gadget.py`), the owner (`owner.py`), the lease modeld borrows the endpoints or
a phone's dial on (`lending.py`), the USB-C port (`port.py`) and the wrapper
that runs this script under `sudo -n`, on AGNOS only (`root.py`).
It is standard library only, so the owner stays at about 10 MB. The zoompilot
fork keeps a shim, `openpilot/sunnypilot/accelerators/jetlink/owner.py`, that
starts it with the fork's provisioning worker.

| Descriptor | Value |
| --- | --- |
| idVendor:idProduct | `1209:0001` (pid.codes test allocation) |
| bcdDevice | `0x0100`; `0x0101` for iOS, so hosts refetch cached descriptors |
| bDeviceClass/SubClass/Protocol | `0x00/0x00/0x00`, class per interface; for iOS `0xEF/0x02/0x01`, Miscellaneous with interface association (composite) |
| Interface 0 | `0xFF/0xFF/0xFF` vendor specific, one bulk IN and one bulk OUT endpoint: the Jetlink link |
| Interfaces 1 and 2 (iOS only) | CDC-NCM control and data: the network for an iPhone |
| Network (iOS only) | comma `192.168.60.1/24`; DHCP `192.168.60.2` to `.254` with 10 minute leases from dnsmasq on the gadget's interface (`usb1` or later, since the modem holds `usb0`), no router or DNS options |

On the Jetson, the installer sets up the server as a service; for a manual run,
from a checkout:

```bash
sudo docker/run.sh --transport usb
```

Jetlink uses the pid.codes test allocation `1209:0001`. Custom distributions
need their own USB product ID.

## Versions and manual rollback

### Choose a version

The installer follows `main` by default. Mac app releases are built from
that branch. The zoompilot fork records the exact Jetlink commit it was tested with
as its `jetlink_repo` submodule, and `main` is kept compatible with the current
`jetson-trt` branch. If the protocol versions differ, the server rejects the
connection and the comma keeps driving on the small model.

To install a release, or the commit the fork records, instead of `main`, pass
it to the installer. Replace `v0.4.0` with a release tag:

```bash
curl -fsSL https://raw.githubusercontent.com/zoompilot/jetlink/v0.4.0/install.sh | bash -s -- --ref v0.4.0
```

### Restore a Docker image

Or put an earlier image back by hand: `sudo docker image ls` shows the images on
the machine, and the one to run is `JETLINK_IMAGE` in `/etc/jetlink/server.env`
(an image ID from `sudo docker image inspect --format '{{.Id}}' IMAGE`). Then
`jetlink restart`. Each update keeps the settings it replaced as
`/etc/jetlink/server.env.prev`, so going back one update is
`sudo cp /etc/jetlink/server.env.prev /etc/jetlink/server.env` and
`jetlink restart`.


## Deep sleep and USB wake

Choosing **Always on** in the installer enables USB wake on the Jetson's hubs,
gives the container access to `/sys/power`, and sets `--sleep-after 120`.
The installer checks for `deep` support in `/sys/power/mem_sleep`.

After ignition off, the comma releases USB once the engine is ready and at
least one minute has passed. The Jetson sleeps after 120 seconds without a
USB connection. Connecting or disconnecting USB wakes it; without a new
connection, it sleeps again after 120 seconds.

If sleep fails, retries start after 10 seconds and double up to 5 minutes.
Check logs, write access to `/sys/power`, and USB wake on the root and onboard
hubs. Without `--sleep-after`, the link stays connected while the comma is awake.

## Battery-protection shutdown

When enabled, the comma requests shutdown at 11.8 V or after 30 hours parked.
The server writes a flag in the models folder; `jetlink-poweroff.path` triggers
the host service to remove it and power off. Flags from earlier boots are ignored.

To test without powering off, run `touch /mnt/data/jetlink/poweroff-dry-run`.
Remove that file to restore shutdown behavior.

Automatic restart after full shutdown requires hardware to cycle DC power or
trigger the J14 power-button input. The devkit boots when DC power returns.
