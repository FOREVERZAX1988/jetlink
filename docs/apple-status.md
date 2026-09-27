# Jetlink on Apple devices: where the work stands (2026-09-27, evening)

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

**The link's medium**: the comma's hello names its link (usb, cable or tcp)
and the speed its USB controller negotiated (`Transport.link_info`); both
servers put it in the link event as `medium`, and both apps show **USB 3**,
**USB 2** or **TCP**, with USB 2 as a warning. Without the new comma code
the server falls back to what it sees itself: the bus speed over USB, TCP
otherwise, and USB of unknown speed for a phone's cable.

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

## Verified on the bench (2026-09-27, evening)

Comma four on the fork's `iphone` branch, Cinque Terre V3, the Mac (M1 Pro) on
a USB 3 C-to-C cable, parked live bench `jetlink_live_bench.sh`:

| Setup | p50 | p99 | Over 50 ms | Drops |
| --- | ---: | ---: | ---: | ---: |
| Python server, USB | 37.69 ms | 48.49 ms | 8 | up to 2.27% |
| Swift server (`jetlink-serve --usb`), USB | 35.61 ms | 38.18 ms | 1 | 0 |
| Swift-only Mac app, USB, screen locked, App Nap fix | 36.4 ms | 40.9 ms | 3 | 0 |
| Swift `--dial` over the comma's cable network (phone stand-in) | 40.4 ms | 49.5 ms | | |

The Mac gate passed, so `mac-swift-only` is ready to merge and release. The
cable row is a whole frame with every frame logged (1,636 frames): comma warp
and readback 4.2 ms, link 8.4 ms (p99 11.2), the Mac's model 27.8 ms. Over USB
the link is about 3.3 ms. The iPhone app ran in the iOS 26.5 simulator, dialed
the comma over the Mac's cable interface and showed the link; it has not run
on a phone.

Found and fixed on the way (fork `iphone`, jetlink `iphone`):
- the comma guessed whether a phone was on the port, holding every Jetson or
  Mac reconnect 5 to 10 s for a dial; the comma's **Accelerator Link** is now
  Off, USB or iOS, USB is the plain gadget lent at once, and iOS never lends
  the endpoint files (fork a40ec1e0da, jetlink 5ab13ec);
- a borrower kept its first loan for the drive, so a phone that dialed later
  was never used; the loan now asks again every attempt;
- a run with nothing to do recorded "the far end sleeps"; it now keeps what
  an earlier run learned, and an iOS gadget never goes dormant;
- after a link loss the first join attempt reused the retired model's closed
  client (EBADF, 5 s lost); a closed client is dead and is replaced;
- over the cable modeld hands the socket the warp's GPU mapping instead of a
  host copy: 0.6 ms off the comma's p50 and 2.2 ms off its p99 (two A/B pairs);
- the comma's kernel gives the host a new MAC every bind, which ran the 8
  address DHCP pool dry; it is the whole subnet now, with 10 minute leases;
- the Swift USB read landed in a copy of the buffer (`NSMutableData(bytesNoCopy:)`
  copies);
- the Mac app's in-process server was throttled by App Nap behind a locked
  screen (p50 75 ms, 45% drops); it holds a latency-critical activity while a
  comma is connected.

Traps: macOS names a new USB network interface only while the screen is
unlocked, and the comma's gadget brings a new MAC every bind, so keep the Mac
awake (`caffeinate -d -u`). Little Snitch held `jetlink-serve`'s traffic on
the cable behind prompts nobody answered: the handshake worked and no data
moved. Each rebuilt binary can prompt again.

## Not verified

1. **Nothing has run on an iPhone.** See [Timing the model on an
   iPhone](#timing-the-model-on-an-iphone) for the procedure.
2. **USB 2 on the cable.** The cost at USB 2 is an estimate (4 to 6 ms a
   frame). Measure it with the Mac stand-in on a USB 2 cable.
3. **C-to-C with an iPhone.** On the other session's Mac bench (comma four,
   M1 Pro, the v0.4.3 app) the comma came up as the device on every plug with
   a USB 3 C-to-C cable, at 5 Gb/s. With the comma's Try.SNK forced off it came
   up as the host on 2 of 7 plugs, and the hold made the Mac the host about
   4.5 s after the plug, at super speed (p50 36.8 / p99 45.9 ms, no drops). A
   USB 2 cable cost about 10 ms a frame and 0.88% dropped frames. Nobody has
   tried an iPhone. Watch the comma's `/sys/class/usbpd/usbpd0/current_pr` and
   `current_dr` and the owner log while plugging in.
4. One odd session: right after the Debug app built the engine itself, with
   about 2 GB of disk free, it served 78 ms a frame until the app restarted;
   the reload served 30.7 ms. Not reproduced.
5. The owner's sleep record is global, not per host: a phone's or a Mac's
   "stays up" outlives it, so a Jetson that sleeps, plugged in later in the
   same boot with its engine ready, stays awake parked.

## Timing the model on an iPhone

The phone's model time (about 18 ms on an iPhone 17 Pro in PR #9) is the
largest term in a frame. On the phone, charging on the hub, Low Power Mode off:

1. **Benchmark tab**, 1 minute then 10 minutes (or launch with `-benchmark N`).
   Record p50, p99, max, frames over 50 ms, the first and last window's p99,
   and the thermal state at start and end.
2. **Live bench.** Connect the comma, then on it, parked:
   `OUTPUT=/data/tmp/phone-<setting>-<n> jetlink_repo/scripts/comma/jetlink_live_bench.sh 180`.
   `summary.json`'s `big` has frames, `exec_p50_p99_p999_max_ms`, `over_50ms`,
   `max_drop_pct`. For a per-frame split (warp, readback, send, reply, server
   time) set `SLOW_FRAME = 0.0` in the fork's `model_state.py` for the run and
   restore it after.
3. **Settings to compare**, alternating A, B, A, B: Neural Engine (whole),
   Neural Engine + GPU, GPU; CPU keep-warm on and off; GPU keep-alive on and
   off. Then `verify_parity.py` on the winner.
4. **Untried, and worth a look:** the whole model on `CPUAndNeuralEngine`
   instead of `ALL` (needs its own device tag and a 32-frame parity check);
   CoreML's compute plan, to see which ops leave the Neural Engine;
   `SpecializationStrategy` Default against FastPrediction on the Neural Engine;
   a back-to-back benchmark, to price the idle between frames.

## The simplify pass (2026-09-27, afternoon)

A review for reuse, simplification, efficiency and altitude, and its fixes:
TCP and USB share one frame reader; the session announces its link once and
owns the medium; the USB gadget reads the pinned IDs and a steady-state read
is one synchronous request; the reply is written after the host's lock is
released; the drain after a desync lives in the USB transport; send helpers,
test clients and test helpers exist once; the registry reads its catalog
once per inventory; log lines reach the Logs view in order through one
stream; the Mac starts its server off the main thread. Skipped, with the
reason in the session: a module split for the Linux build, one structured
log format for both apps, larger USB reads (the Python host's read sizes are
pinned), and moving the model output straight into the USB send buffer.

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

The comma is on the fork's `iphone` branch with `DisableUpdates=1`, for more
bench work; restore it with step 5 below. The Jetson runs its normal service.
The comma is on the Mac's USB 3 C-to-C cable.

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
