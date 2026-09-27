# Jetlink on Apple devices: where the work stands (2026-09-27)

Pickup notes for the iPhone app, the Mac's Swift server and the comma-side
changes behind them. Read this before touching any of the branches below.
It replaces `iphone-cable-link-status.md`.

## Branches and worktrees

| Repo | Branch | Where | State |
| --- | --- | --- | --- |
| jetlink | `iphone` | `../jetlink-iphone`, pushed to `zoom` | the iPhone app, the Swift server as the Mac's USB host, the Mac's Server setting and Benchmark page, the conformance suite (merged from `conformance`) |
| jetlink | `conformance` | `../jetlink-conformance`, pushed to `zoom` | merged into `iphone`; CI green on macOS and Linux. Kept only as the record of that work |
| jetlink | `mac-swift-only` | `../jetlink-mac-swift`, pushed to `zoom` | stacked on `iphone`, NOT merged: the Mac app with the Swift server only and no embedded Python. Merge it only after the USB gate below passes |
| zoompilot fork | `iphone` | `../sunnypilot-iphone`, pushed to `zoom` | the phone's dial and the cable link, plus the USB-C port hold picked from the `usbc-device-role` work (fork 1fd0f8d234, e79072d594, 0673caf11a, f216abb5d2 on `zoom/danger-unstable`); `jetlink_repo` pinned to the jetlink `iphone` tip |
| jetlink | `pr-9` | local only | the other contributor's iPhone PR, for reference |

`main` and `jetson-trt` are untouched by all of this.

## What each side has

**Swift (`JetlinkKit`)**, shared by both apps:
- The server (`JetlinkServer`): wire protocol, session, queues, the V3 state
  loop, onnxruntime's CoreML provider, the benchmark, CPU keep-warm, and
  `EmbeddedServer`, the in-process server both apps drive.
- Transports behind `MessageLink`: TCP (listen, or dial the comma over the
  cable), and USB host through IOUSBHost on a Mac (`USBTransport` for the
  framing, `USBGadget` for the device). The USB framing follows the Python
  host's rules: each gadget message read to its 16 KB boundary in whole
  packets, at most 256 KB a read, one transfer per sent message with the
  one-byte PADDED rule, a latched desync and a drain. A gadget on the bus that
  nothing on the comma is serving yet fails its first read; that session
  reports no link and is retried quietly (0.5 s for five tries, then 2 s).
- `JetlinkONNX` (the preparation, byte-identical to Python for split, whole and
  ane-whole), `JetlinkRegistry`, `JetlinkUI` (the frame budget, the benchmark
  verdict and windows, the log colours).

**Mac app** (`macos/`, on `iphone`): Settings > Server picks Python (the
default) or Swift. Swift runs `EmbeddedServer` in process over USB or TCP, logs
in the Python server's format to the Logs view and `server.log`, and has a
Benchmark page. Automatic maps to the Neural Engine split, CoreML on the GPU to
the GPU; tinygrad is Python only.

**iPhone app** (`ios/`): unchanged in behaviour this session; it now drives
`EmbeddedServer` and takes its benchmark pieces from `JetlinkUI`.

**Fork** (`openpilot/sunnypilot/accelerators/jetlink/`): the phone dials the
owner on 192.168.60.1:5599 over the composite gadget's network link, the loan
carries the socket, and (new) `usbport.Port` holds the comma's USB-C port as
the device for any host that is not a chestnut, so a C-to-C host (Mac or
iPhone) gets the host role. It forces the charger's DISABLE_POWER_ROLE_SWITCH
voter and leaves USB PD alone (fork f216abb5d2). See the fork commits.

## Verified, and how

- Swift USB transport, against a fake bulk pair (no comma was on the Mac's
  USB): gadget framing including 16 KB boundaries and a 4 MB message, read
  sizes, partial and stalled writes, desync and drain, shutdown, the quiet
  retry of an unserved gadget with no link events, and the golden frames of
  both tiny models served bit for bit through the USB framing.
- Mac app on the Swift server, over TCP loopback: served Cinque Terre V3 from
  `bench_link.py` (400 frames, none over 50 ms, Debug build), Status shows the
  server, the Benchmark page renders with the model and its buttons.
- Python server against Swift server, same Mac and model, loopback, release
  builds: Swift about 1 ms faster a frame at p50, both 0 over 50 ms on clean
  runs. Table in [mac-performance.md](mac-performance.md#the-python-server-and-the-swift-server).
- One cache for both servers: the Python server loaded an engine the Swift
  server built (converted model already deleted) and served 29.7 ms p50.
- Conformance ([conformance.md](conformance.md)): the Python at HEAD writes
  every fixture and `Pinned.swift`, CI regenerates and diffs them, pytest
  checks Python against them (456 passed) and Swift reads them (227 tests on
  macOS; the portable modules also build and pass on Linux in `swift:6.2-noble`,
  CI job "Swift conformance on Linux"). One drift found and fixed: the Swift
  rounded published numbers half away from zero where Python rounds to even.
  `PREPARE_VERSION` is 5 on both sides; the Swift's 6 only forced a rebuild at
  every switch between servers. swift-crypto stands in for CryptoKit in the
  Linux build only.
- Fork suites on the `iphone` branch after the USB-C picks: 365 passed.
- Earlier (2026-09-27 morning, Jetson as the host): USB 28.59 / 30.31 ms p50 /
  p99, cable link 36.43 / 40.55 ms with the Linux host's NCM block capped at
  2 KB. The Jetson recipe, the relay and the traps are in the git history of
  this page (`iphone-cable-link-status.md` at 687a5da).

## Not verified

1. **The Swift server as the Mac's USB host has never met a comma.** Nothing was
   plugged into the Mac. Unknowns: IOUSBHost opening the vendor interface of
   the composite gadget, what an unserved gadget's first read returns on macOS,
   re-enumeration during a handover, and the latency.
2. **The Mac gate for step 4.** Python server against Swift server over USB,
   parked, `jetlink_repo/scripts/comma/jetlink_live_bench.sh 180` each, same
   cable and model. Pass: Swift's p99 no worse and no frame dropped. Then merge
   `mac-swift-only` and release it as its own version.
3. **Nothing has run on an iPhone.**
4. **C-to-C with an iPhone.** The other session's Mac bench (comma four, M1 Pro)
   found the comma comes up as the device on every plug with a USB 3 C-to-C
   cable, at 5 Gb/s, so the hold never fired; three cables whose e-markers say
   USB 2 only stayed at 480 Mb/s. Nobody has tried an iPhone. Watch the comma's `/sys/class/usbpd/usbpd0/current_pr` and
   `current_dr` and the owner log while plugging in.
5. Latency parity over the cable link on a Linux host (+8 ms a frame).
6. One odd session: right after the Debug app built the engine itself, with
   about 2 GB of disk free, it served 78 ms a frame until the app restarted;
   the reload served 30.7 ms. Not reproduced.

## Two things learned the hard way

- **A full disk makes CoreML fall back to the CPU without an error.** The
  Neural Engine compile writes a bundle cache under `~/Library/Caches/<process>`;
  with no space it fails (`ANECCompile() FAILED` on stderr only) and the model
  runs at about 375 ms a frame, which the comma sees as a dead link. The disk
  was full twice this session. A server-side check (a proving run slower than
  a budget, or free space below a few GB before a load) would turn this into a
  message; neither server has one.
- `~/Library/Caches/python3/com.apple.e5rt.e5bundlecache` had grown to 4 GB of
  Neural Engine compiles from benches since 2026-09-09. It is a cache; stale
  entries can go.

## Bench state after this session

Untouched: the comma is on `danger-unstable` with `DisableUpdates=0` (it will
pick up the USB-C port hold, which landed on `zoom/danger-unstable`), and the
Jetson runs its normal service. Nothing is cabled to the Mac.

## How to bench the Mac over USB

1. On the comma: `git fetch origin iphone && git checkout iphone`, set
   `DisableUpdates=1`, then from `../jetlink-iphone` run
   `scripts/deploy_to_comma.sh comma@192.168.1.144` and reboot. (Any branch
   whose `jetlink_repo` has the composite gadget works; the plain gadget works
   too, the Swift host falls back to interface 0.)
2. Unplug the comma from the Jetson and plug it into the Mac: a USB-A port on a
   hub or adapter with an A-to-C cable, or a USB 3 C-to-C cable.
3. Mac app, Settings > Server > Python, Restart Server; wait for the link; on
   the comma, parked, run the live bench for 180 s. Then Swift, restart, the
   same. Record both in `mac-performance.md`.
4. Or from a checkout: `JetlinkKit/.build/release/jetlink-serve --cache
   "$HOME/Library/Application Support/Jetlink/cache" --usb` (quit the app
   first: only one process can hold the interface).
5. Restore: the comma back to `danger-unstable` with `DisableUpdates=0`, the
   cable back to the Jetson.
