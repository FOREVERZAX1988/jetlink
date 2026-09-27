# iPhone over one cable: where the work stands (2026-09-27)

Pickup notes for the `iphone` branches. Read this before touching either branch.

## Branches and worktrees

| Repo | Branch | Where | State |
| --- | --- | --- | --- |
| jetlink | `iphone` | `../jetlink-iphone`, pushed to `zoom` | 53 commits over main 92d3cd7, CI green |
| zoompilot fork | `iphone` | `../sunnypilot-iphone`, pushed to `zoom` | 6 commits over jetson-trt 1c85c44cb; submodule `jetlink_repo` pinned to jetlink `iphone` 5e4d93a (three doc-only commits behind the tip) |
| jetlink | `pr-9` | local only | the other contributor's iPhone PR (github.com/zoompilot/jetlink/pull/9), for reference |

The design and the review of PR #9 are in the session plan that produced this
branch; the short form is in `docs/transport.md` and `docs/iphone-app.md`.

## What is on the jetlink branch

- `scripts/setup_gadget.sh`: one composite gadget, `ffs.jetlink` first (interface 0, so
  every existing host keeps working), then `ncm.usb0` (ECM fallback; this kernel has no
  ECM, RNDIS or EEM), device class EF/02/01, bcdDevice 1.01. `--net` (run by the fork's
  owner after every bind, because the netdev only exists while the gadget is bound)
  configures the comma's end: the netdev is NOT `usb0` (the modem owns that name), read
  `functions/ncm.usb0/ifname`; NetworkManager unmanaged; 192.168.60.1/24; dnsmasq DHCP
  .2-.9 with no router or DNS option; status in `/dev/shm/jetlink-net`.
- `jetlink/transport/usbbulk.py`: the host finds the vendor interface by class FF/FF/FF,
  not by number 0. `jetlink/client.py`: `open_socket(sock)`. `bench_link.py` and
  `verify_parity.py`: `--listen` accepts a phone's dial.
- `JetlinkKit` (Swift): dial mode (`Server.setDial`, `jetlink-serve --dial HOST:PORT`),
  CPU keep-warm, no spinning in onnxruntime's thread pools, vImage fp16 conversions, the
  MLProgram `Data` dir dropped after compile, self-healing listener, `stop()` keeps the
  engine and `shutdown()` releases it, the in-app benchmark (`ControlCommand.benchmark`),
  `LogBuffer` shared with the Mac, a log sink. `JetlinkONNX`: the ane-whole layout
  (`prescaleLayerNorm`, `visionHeads`, `headsInFP32`), byte-identical to
  `jetlink/onnx_patch.py` (fixtures under `Tests/JetlinkONNXTests/Fixtures`, 54 cases).
  `CoreMLBackend.Device.aneWhole`, prepare version 6. Python: `--device ane-whole`.
- `ios/`: dials 192.168.60.1:5599 whenever a 192.168.60.x interface exists, Benchmark
  tab, Logs screen, memory and thermal tiles, the stay-on-screen banner, the shutdown
  alert, `Config/Local.xcconfig` for signing, Release run scheme, `-tab` and
  `-benchmark N` launch arguments for screenshots. Default compute: Neural Engine
  (ane-whole).
- Docs: `docs/iphone-app.md`, `docs/transport.md` (the network link section, the Linux
  host block-size rule), `scripts/99-jetlink-host.rules`.

## What is on the fork branch

`openpilot/sunnypilot/accelerators/jetlink/`: `gadget.link_kind()` (usb, cable, ethernet),
`/dev/shm/jetlink-link`, `net_up()`, `usb_speed()`; `lending.CableListener` (the owner
accepts the phone's dial on 192.168.60.1:5599, one held socket, `send_fds` in the loan,
`Loan.sock`); the owner holds borrowers off FunctionFS for `CABLE_HOLD` (5 s) after the
UDC reaches configured, never goes dormant or bounces over the cable, spawns runs over
TCP; `wait_for_host` and `borrow` return at once over TCP; no bounce over TCP; the UI
link status names the transport. Tests: 321 in the package.

## What was verified, and how (bench mici .144 + Jetson jetlink.local as the host)

- USB baseline, live bench 180 s: big p50 28.59 / p99 30.31 / max 44.1 ms, 0 over 50.
- Composite gadget: super-speed, the old Jetson server image connects on interface 0,
  cdc_ncm binds interface 1, the Jetson takes a DHCP lease, the owner logs
  `cable link from 192.168.60.x` when a dial arrives, lends the socket to modeld, runs
  a provisioning run over it, and modeld ran a full live bench over the cable link.
- The Jetson stood in for the phone: server in TCP mode
  (`sudo JETLINK_ENV_FILE=/tmp/server-tcp.env /usr/local/lib/jetlink/run-server` with
  TRANSPORT=tcp and SLEEP_AFTER=0 in that file; the env file is sourced after the
  environment so plain env vars do not work) plus a relay that dials the comma and
  bridges to 127.0.0.1:5599 (a 60-line python script; rewrite from `docs/transport.md`
  if needed: connect, retry 0.5 s, pump both ways, redial when a side closes).
- Cable link, live bench 180 s, host `cdc_ncm/rx_max` 2048: big p50 36.43 / p99 40.55 /
  max 76.5, 2 over 50, 0 drops. `bench_link.py --host` from the comma: 43 ms round trip,
  26 ms transport vs 8.6 ms over FunctionFS, all of it the comma's send of 393 KB.
- The Linux host must cap `rx_max` to 2048: at the 16 KB default the 4.9 gadget sends
  22 Mbit/s (169 ms round trip, 71% drops). The gadget cannot advertise a smaller block;
  u_ether `tx_qmult` does not help. The udev rule in `scripts/99-jetlink-host.rules`
  applies the cap on Linux hosts. Apple's driver is unmeasured.

## Not done

1. Nothing has run on an iPhone: the app ran in the simulator only (screenshots were
   taken there). The first phone test: app open, USB 3 hub, USB 3 A-to-C cable into
   the comma; watch `/data/log/jetlink-owner.log` for `cable link from`, then
   `tools`: `jetlink_repo/scripts/comma/jetlink_live_bench.sh 180` on the comma.
2. The Mac has not been plugged into the composite gadget.
3. Latency parity was not reached on a Linux host (+8 ms per frame); the phone number
   decides whether that matters. The escape hatch in the plan (UDP framing) is unbuilt.
4. The fork's submodule pin can move to the jetlink tip; `jetson-trt` and `main` are
   untouched by all of this.

## Bench state after this session

The comma is back on `danger-unstable` 1c85c44cb with the main jetlink package
(v0.4.0, plain gadget) rsync'd into `jetlink_repo`, `DisableUpdates=0`. The Jetson runs
its normal service (USB, sleeps). To bench the branches again: on the comma
`git fetch origin iphone && git checkout iphone`, set `DisableUpdates=1`, then from
`../jetlink-iphone` run `scripts/deploy_to_comma.sh comma@192.168.1.144` (its excludes
now leave the Swift trees and build products behind; the first run shipped 2 GB), reboot
the comma. Reverse: `git checkout danger-unstable`, rsync the main package, reboot.
