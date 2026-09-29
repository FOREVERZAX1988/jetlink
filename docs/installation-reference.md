# Installation reference

Normal setup: [Jetson guide](jetson.md), [Mac app](macos-app.md),
[PC guide](platforms.md). This page: what the installer does, manual installs
and custom integrations.

<a id="jetson-installation"></a>

## Jetson and PC installation

### What the installer changes

- Installs NVIDIA's TensorRT libraries if missing: `libnvinfer10` and
  `libnvonnxparsers10` from JetPack's package source (on JetPack 7.2 the
  newest, at least 10.16.2.10), or on a PC `libnvinfer11` and
  `libnvonnxparsers11` 11.3.0.99 from NVIDIA's CUDA package source.
- Unpacks the release's server to `/opt/jetlink/<version>`, with
  `/opt/jetlink/current` pointing at it and `previous` at the one before, and
  checks it can use the GPU before it replaces the running one.
- Installs the `jetlink-server` service (runs as root) and the `jetlink`
  command. Settings: `/etc/jetlink/server.env`; your answers:
  `/etc/jetlink/install.conf`.
- Serves the read-only status page on port 5600 (a question; 0 turns it off).
- Keeps models and prepared engines in `/mnt/data/jetlink` on a Jetson,
  `/var/lib/jetlink` on a PC.
- Jetson only: sets the fastest power mode (MAXN SUPER on an Orin Nano; may
  need one restart) and runs `jetson_clocks` before every start; adds 8 GB of
  swap for preparing the 1.7 GB models; stops boot waiting for a network (none
  in the car; about two minutes); caps the system log at 200 MB.

An install from 0.6.0 or earlier runs the server in Docker. Its next
`jetlink update` moves it to the native server, keeping the answers, models,
engines and Jetson setup, and saves the Docker setup in
`/etc/jetlink/docker-era`. Jetlink's Docker images go once the new server runs;
Docker itself stays.

### Installing by hand

The installer is the supported path. Its pieces, from a release tarball:

1. TensorRT, as above.
2. The tarball unpacked to `/opt/jetlink/<version>`, and
   `sudo ln -sfn /opt/jetlink/<version> /opt/jetlink/current`.
3. `share/jetlink/systemd/jetlink-server.service` in `/etc/systemd/system`,
   and `/etc/jetlink/server.env` with `JETLINK_CACHE_DIR`,
   `JETLINK_SLEEP_AFTER` and `JETLINK_STATUS_PORT` (the unit's defaults:
   `/var/lib/jetlink`, `0`, `5600`). Then
   `sudo systemctl enable --now jetlink-server`.
4. Jetson: a drop-in for the service with `ExecStartPre=-/usr/bin/jetson_clocks`.
5. Always-on supply only (lets the comma wake the Jetson):
   `share/jetlink/udev/99-jetlink-usb-wakeup.rules` in `/etc/udev/rules.d`.

`jetlink run` runs the server in the terminal instead of the service (Ctrl-C
stops it); extra flags pass through, such as `--listen` for a TCP bench.

## Custom USB integrations

- comma 3X (AGNOS kernel 4.9.103) has FunctionFS and USB gadget support.
- The server is always the USB host, through usbfs (IOKit on a Mac): no driver
  and no gadget kernel modules on the host.
- On Linux the server turns off USB 3 link power management (U1/U2) on the
  comma's port each time it claims it: 3.9 of the 7.6 ms transport on the
  bench Jetson. Deep sleep and USB wake are unaffected. `JETLINK_USB_LPM=1` in
  the service's environment leaves it on.
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

- The installer installs the newest release; `jetlink update` moves to the
  next. `jetlink update --ref main` switches to development builds (the `edge`
  prerelease).
- The zoompilot fork pins the Jetlink it was tested with as its `jetlink_repo`
  submodule.
- The comma and the server speak one protocol (protocol 3 since 0.7.0): update
  them together. On a mismatch the server refuses the comma, which drives on
  its small model, and the logs name the side to update:
  `update the comma's jetlink package` or `update jetlink on the Jetson`.

To pin a release, pass it to the installer (replace `v0.7.0`);
`jetlink update --ref latest` follows releases again:

```bash
curl -fsSL https://raw.githubusercontent.com/zoompilot/jetlink/v0.7.0/install.sh | bash -s -- --ref v0.7.0
```

<a id="restore-a-docker-image"></a>

### Roll back by hand

- Back one update from 0.7.0 or later: point `current` at `previous` and
  restart. `jetlink update` moves forward again.

  ```bash
  sudo ln -sfn "$(readlink -f /opt/jetlink/previous)" /opt/jetlink/current
  jetlink restart
  ```

- Back to 0.6.0, the last Docker release: `jetlink update --ref v0.6.0`. Its
  own installer takes over with the same answers. 0.6.0's `jetlink update`
  cannot install a native release, so to come forward again run:

  ```bash
  curl -fsSL https://raw.githubusercontent.com/zoompilot/jetlink/main/install.sh | bash -s -- --update --ref latest
  ```

## Deep sleep and USB wake

**Always on** in the installer installs a udev rule that lets the USB hubs wake
the Jetson, sets `--sleep-after 120`, and checks for `deep` in
`/sys/power/mem_sleep`. The server arms the hubs again before every suspend.

- Ignition off: the comma releases USB once the engine is ready and at least
  one minute has passed.
- The Jetson sleeps after 120 s without a USB connection. USB connect or
  disconnect wakes it; with no new connection it sleeps again after 120 s.
- `jetlink caffeinate` keeps an awake Jetson awake, like the Mac's
  `caffeinate`: until Ctrl-C, for `-t SECONDS`, or while `COMMAND` runs. No
  sudo needed. Updates hold it awake on their own.
- Failed sleep retries after 10 s, doubling up to 5 minutes. Check the logs and
  USB wake on the root and onboard hubs.
- With `--sleep-after 0`, the link stays up while the comma is awake.

## Battery-protection shutdown

- When enabled, the comma requests shutdown at 11.8 V or after 30 hours parked.
- The server answers, syncs and powers the Jetson off (`systemctl poweroff`). A
  PC stays up.
- A file named `poweroff-dry-run` in the models folder keeps it up. The
  installer writes it when you answer No; `touch
  /mnt/data/jetlink/poweroff-dry-run` does it by hand, and removing the file
  restores shutdown.
- Restart after full shutdown needs hardware that cycles DC power or triggers
  the J14 power-button input; the devkit boots when DC power returns.
