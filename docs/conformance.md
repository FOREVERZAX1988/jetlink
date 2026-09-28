# Keeping the Swift and Python servers in step

A comma must not be able to tell the two servers apart, and one model cache
must serve both.

| Server | Code | Runs |
| --- | --- | --- |
| Python | `jetlink/server` | Jetson, Linux PC, and the Mac app until the Swift one takes over |
| Swift | `JetlinkKit/Sources/JetlinkServer` | the iPhone app, the Mac app behind a setting, `jetlink-serve` |

Python is the source: what the two must agree on is generated from it,
committed, and read by the Swift tests; the Python tests check the Python still
writes exactly those files.

## What is pinned

| What | Written by | Read by the Swift in |
| --- | --- | --- |
| Constants: the wire's magic, version, sizes, message, flag and status numbers; USB ids and packet sizes; model constants; control protocol version; `PREPARE_VERSION`; the onnxruntime release the Apple builds link | `make_pins.py` writes `JetlinkKit/Sources/JetlinkKit/Pinned.swift` | the Swift code uses `Pinned`; `ConformanceTests` checks every Swift constant against it |
| Wire bytes: headers and INFER bodies; the byte streams of TCP, a USB host and the gadget's 16 KB bursts; the reads a USB host posts | `make_conformance_fixtures.py wire` | `JetlinkServerTests/ConformanceTests.swift` |
| Tensors the queues stage each frame at frame_skip 1, 2 and 4, a reset included | `make_conformance_fixtures.py staging` | the same file |
| The `stats` event from fixed samples | `make_conformance_fixtures.py stats` | the same file |
| Every control-channel event a real `ControlServer` writes over a real registry and cache | `make_conformance_fixtures.py control` | `JetlinkKitTests/PythonControlEventsTests.swift` |
| LFS pointers, model identities, catalog parsing and merging, one cache directory's catalog and inventory payloads | `make_conformance_fixtures.py registry` | `JetlinkRegistryTests/ConformanceTests.swift` |
| ONNX preparation for the split, whole and ane-whole layouts, byte for byte, on graphs that take every branch | `make_onnx_fixtures.py` | `JetlinkONNXTests/PreparationTests.swift` |
| Whole-server runs: driving output of the tiny queued and stateful graphs, bit for bit, over TCP and USB | `make_server_fixtures.py` | `JetlinkServerTests/ServerTests.swift` and `USBTransportTests.swift` |

- Generators live in `JetlinkKit/Scripts`. Each imports the `jetlink` package
  of its own checkout, whatever the environment has installed.
- Published numbers round as Python's `round()` does (half to even on the exact
  binary value), through `pythonRound` in JetlinkKit.

## How it runs

- `tests/test_conformance.py` reruns every generator into a temporary directory
  and compares byte for byte with the committed files. It also checks that
  `Pinned.swift` is current, the Swift package links the pinned onnxruntime,
  and CI regenerates with the pinned releases.
- `swift test --package-path JetlinkKit` reads the fixtures on macOS (`swift`
  CI job) and Linux (`swift-linux`, below).
- `python-test-macos` runs that test on an Apple arm64 runner with the pinned
  releases installed, so no comparison skips.

Tool-dependent files:

| Files | Made by | Compared only |
| --- | --- | --- |
| ONNX graphs | onnx's serialiser | under the onnx release that made them |
| Golden outputs | onnxruntime's CPU provider | on Apple arm64, under the release the Swift package links |

Both releases, with the numpy and protobuf they ran with, are in
`JetlinkKit/Scripts/fixture-pins.txt`, which the test, `make_pins.py` and CI
read. Elsewhere those comparisons skip.

Real models are too big for fixtures. `JetlinkKit/Scripts/check_onnx_prep.py`
compares the two preparations on any ONNX, in any layout
(`--layout split|whole|ane-whole`, or `--all`), and runs both chains on
onnxruntime for graphs under 64 MB. The three models in the Mac app's cache,
the comma's (a086d5249fc3) among them, came out byte-identical in ane-whole.

## When a change is intentional

1. Change the Python.
2. Regenerate what moved, from the checkout root:

   ```bash
   .venv/bin/python JetlinkKit/Scripts/make_pins.py
   .venv/bin/python JetlinkKit/Scripts/make_conformance_fixtures.py   # or one part: wire, staging, stats, control, registry
   .venv/bin/python JetlinkKit/Scripts/make_server_fixtures.py
   .venv/bin/python JetlinkKit/Scripts/make_onnx_fixtures.py
   ```

   The venv needs the pinned onnx, onnxruntime, numpy and protobuf
   (`pip install -r JetlinkKit/Scripts/fixture-pins.txt`) for the ONNX and
   output files to match.
3. Change the Swift until `swift test --package-path JetlinkKit` passes.
4. Commit the Python, fixtures and Swift together.

A change to what a CoreML build writes bumps `PREPARE_VERSION` in
`jetlink/server/backends/ort/__init__.py`, which both servers read. An artifact
prepared under another version rebuilds on its next load.

## One cache for both servers

- Both write the same artifact under the same name: the model's identity, the
  onnxruntime release and the CoreML device in the tag; the prepared ONNX and
  onnxruntime's compiled model inside.
- ane-whole has its own device tag, so it never collides with split.
- The Swift reads the prepare version (5) from `Pinned`. Diverging versions
  invalidate nothing, but a Mac switching servers would rebuild every engine on
  every switch.
- Artifacts are interchangeable: the Python server loaded a Swift-built engine
  (relabelled to prepare 5, without the MLProgram `Data` directories Swift drops
  after compiling) and served the comma's model at 29.7 ms p50, 0 of 390 frames
  over 50 ms.

## Linux

The Jetson stays on the Python server (the Swift one has no TensorRT backend).
The Swift package builds on Linux only as the drift check's second platform,
never for deployment. `Package.swift`'s Linux branch builds JetlinkKit,
JetlinkONNX, JetlinkRegistry and the portable part of JetlinkServer; the few
tests that need Apple's frameworks sit behind `canImport(Darwin)` or
`canImport(COrt)`.

| | |
| --- | --- |
| In | wire protocol, TCP transport, USB framing (not the IOUSBHost gadget), queues, conversions (element loops instead of vImage), model spec, frame statistics, ONNX preparer |
| Out | onnxruntime, CoreML, Metal, the session, engine host, server loop, JetlinkUI |
| Stand-ins | swift-crypto for CryptoKit; `JetlinkLog` takes `os.Logger`'s calls; the capped HTTP read fetches the whole body and cuts it (Linux URLSession has no byte stream; this build never fetches a model) |
| Tests | the conformance suites, all of JetlinkONNX's (byte-for-byte preparation included), control protocol and model row tests, and the wire, USB framing, conversion, spec, JSON and cache layout tests: 123 tests in 19 suites. Registry network tests and everything that runs the server loop stay on macOS. |

CI runs it in the `swift:6.2-noble` container. Locally, from the checkout root,
building in memory rather than in Docker's disk image:

```bash
docker run --rm -v "$PWD":/src:ro --tmpfs /work:exec,size=6g swift:6.2-noble bash -c \
  'tar -C /src --exclude=.build -cf - JetlinkKit tests/fixtures | tar -C /work -xf - && swift test --package-path /work/JetlinkKit'
```

## What the Swift refuses on purpose

The Swift preparation has no shape or type inferrer. Where an export omits a
shape or type, Python infers it and Swift does not:

| Fixture | Layout | Swift |
| --- | --- | --- |
| `noshape.onnx` | split | refuses the layout, saying what is missing |
| `notype.onnx`, `noentry.onnx` | ane-whole | refuses the layout, saying what is missing |
| `unrecorded.onnx` | whole, ane-whole | leaves a Gather index Python rewrites, so the files differ |

The Swift tests hold each to the Swift behaviour. The driving models checked so
far record every shape and type these passes read.
