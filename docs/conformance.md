# Keeping the Swift and Python servers in step

Jetlink has two servers. The Python one (`jetlink/server`) runs on a Jetson, a
Linux PC and, until the Swift one takes over, inside the Mac app. The Swift one
(`JetlinkKit/Sources/JetlinkServer`) runs inside the iPhone app, behind a
setting in the Mac app, and as `jetlink-serve`. A comma must not be able to
tell them apart, and one model cache must serve both.

Python is the source. Everything the two must agree on is written down from
the Python's own code, committed, and read by the Swift tests. The Python tests
then check that the Python still writes exactly those files, so neither side
can move alone.

## What is pinned

| What | Written by | Read by the Swift in |
| --- | --- | --- |
| Constants: the wire's magic, version, sizes, message, flag and status numbers; the USB ids and packet sizes; the model constants; the control protocol version; `PREPARE_VERSION`; the onnxruntime release the Apple builds link | `make_pins.py` writes `JetlinkKit/Sources/JetlinkKit/Pinned.swift` | the Swift code itself uses `Pinned`, and `ConformanceTests` checks every Swift constant against it |
| Wire bytes: headers and INFER bodies; the byte streams TCP, a USB host and the gadget's 16 KB bursts produce; the reads a USB host posts | `make_conformance_fixtures.py wire` | `JetlinkServerTests/ConformanceTests.swift` |
| The tensors the queues stage each frame, at frame_skip 1, 2 and 4, a reset included | `make_conformance_fixtures.py staging` | the same file |
| The `stats` event made from fixed samples | `make_conformance_fixtures.py stats` | the same file |
| Every control-channel event a real `ControlServer` writes over a real registry and cache | `make_conformance_fixtures.py control` | `JetlinkKitTests/PythonControlEventsTests.swift` |
| LFS pointers, model identities, catalog parsing and merging, the catalog and inventory payloads of one cache directory | `make_conformance_fixtures.py registry` | `JetlinkRegistryTests/ConformanceTests.swift` |
| The ONNX preparation for the split, whole and ane-whole layouts, byte for byte, on graphs that take every branch | `make_onnx_fixtures.py` | `JetlinkONNXTests/PreparationTests.swift` |
| Whole-server runs: the driving output of the tiny queued and stateful graphs, bit for bit, over TCP and over USB | `make_server_fixtures.py` | `JetlinkServerTests/ServerTests.swift` and `USBTransportTests.swift` |

The generators live in `JetlinkKit/Scripts`. Each imports the `jetlink`
package of the checkout it sits in, whatever the environment has installed.

Numbers the servers publish are rounded as Python's `round()` does, half to
even on the exact binary value, through `pythonRound` in JetlinkKit. The
Swift once rounded halves away from zero and wrote 1.13 where Python wrote
1.12.

## How it runs

- `tests/test_conformance.py` runs every generator again into a temporary
  directory and compares the result with the committed files, byte for byte.
  It also checks that `Pinned.swift` is current, that the Swift package links
  the pinned onnxruntime, and that CI regenerates with the pinned releases.
- `swift test --package-path JetlinkKit` reads the fixtures on macOS (the
  `swift` CI job) and on Linux (the `swift-linux` job, below).
- The `conformance-fixtures` CI job regenerates every fixture on an Apple
  arm64 runner with the pinned releases and fails if `git status` shows any
  change.

Two kinds of file depend on tools as well as on this code. onnx serialises
the graphs, so those files are compared only under the onnx release that made
them (`FIXTURE_ONNX` in `tests/test_conformance.py`, 1.22.0). onnxruntime's
CPU provider computes the golden outputs, so those are compared only on Apple
arm64 under the release the Swift package links (`APPLE_ONNXRUNTIME` in the
ort backend, 1.29.0). Elsewhere those comparisons skip; the
`conformance-fixtures` job installs exactly those releases, so it never skips.

Real models are too big for fixtures. `JetlinkKit/Scripts/check_onnx_prep.py`
compares the two preparations on any ONNX, in any layout
(`--layout split|whole|ane-whole`, or `--all`), and runs both chains on
onnxruntime for graphs under 64 MB. On 2026-09-27 the three models in the
Mac app's cache, the comma's (a086d5249fc3) among them, came out
byte-identical in the ane-whole layout.

## When a change is intentional

1. Change the Python.
2. Regenerate what moved, from the root of the checkout:

   ```bash
   .venv/bin/python JetlinkKit/Scripts/make_pins.py
   .venv/bin/python JetlinkKit/Scripts/make_conformance_fixtures.py   # or one part: wire, staging, stats, control, registry
   .venv/bin/python JetlinkKit/Scripts/make_server_fixtures.py
   .venv/bin/python JetlinkKit/Scripts/make_onnx_fixtures.py
   ```

   The venv needs the pinned onnx, onnxruntime, numpy and protobuf
   (`FIXTURE_PINS` in `.github/workflows/ci.yml`) for the ONNX and output
   files to come out the same.
3. Change the Swift until `swift test --package-path JetlinkKit` passes.
4. Commit the Python, the fixtures and the Swift together.

A change to what a CoreML build writes bumps `PREPARE_VERSION` in
`jetlink/server/backends/ort/__init__.py`, which both servers then read.
An artifact prepared under another version is rebuilt on its next load.

## One cache for both servers

Both servers write the same artifact under the same name: the model's
identity, the onnxruntime release and the CoreML device in the tag, the
prepared ONNX and onnxruntime's compiled model inside. The ane-whole layout
has a device tag of its own, so it never collides with the split one.

The Swift server had moved its prepare version to 6 for ane-whole while the
Python stayed at 5. That invalidated nothing, but a Mac switching between the
two servers would have rebuilt every engine on every switch. The Swift now
reads 5 from `Pinned`.

It is safe because the artifacts are interchangeable. A Swift-built engine,
with the MLProgram `Data` directories it drops after compiling, was
relabelled to prepare 5 in a scratch cache. The Python server loaded it and
served the comma's model at 29.7 ms p50, with none of 390 frames over 50 ms.

## Linux

The Jetson stays on the Python server. Its TensorRT path is the one proven on
the bench and in the car, and the Swift server has no TensorRT backend.

The Swift package builds on Linux only as the drift check's second platform,
never as a deployment. `Package.swift` has a Linux branch that builds
JetlinkKit, JetlinkONNX, JetlinkRegistry and the portable part of
JetlinkServer; the few tests that need Apple's frameworks sit behind
`canImport(Darwin)` or `canImport(COrt)`:

- in: the wire protocol, the TCP transport, the USB framing (not the IOUSBHost
  gadget), the queues, the conversions (element loops in place of vImage),
  the model spec, the frame statistics and the ONNX preparer;
- out: onnxruntime, CoreML, Metal, the session, the engine host and the
  server loop, and JetlinkUI;
- swift-crypto stands in for CryptoKit, and `JetlinkLog` gives `os.Logger`'s
  calls somewhere to go;
- the tests are the conformance suites, all of JetlinkONNX's (the
  preparation byte for byte among them), the control protocol and model row
  tests, and the wire, USB framing, conversion, spec, JSON and cache layout
  tests: 123 tests in 19 suites. The registry's network tests and everything
  that runs the server loop stay on macOS.
- the capped HTTP read fetches the whole body and cuts it, because Linux's
  URLSession has no byte stream; this build never fetches a model.

CI runs it in the `swift:6.2-noble` container. The same thing locally, from
the root of the checkout, with the build in memory rather than in Docker's
disk image:

```bash
docker run --rm -v "$PWD":/src:ro --tmpfs /work:exec,size=6g swift:6.2-noble bash -c \
  'tar -C /src --exclude=.build -cf - JetlinkKit tests/fixtures | tar -C /work -xf - && swift test --package-path /work/JetlinkKit'
```

## What the Swift refuses on purpose

The Swift preparation has no shape or type inferrer. Where an export leaves
out a shape or a type, Python infers it and Swift does not:

- `noshape.onnx` (split) and `notype.onnx` and `noentry.onnx` (ane-whole):
  Swift refuses the layout with a message saying what is missing.
- `unrecorded.onnx` (whole and ane-whole): Swift leaves a Gather index that
  Python rewrites, so the files differ.

The Swift tests hold each of these to the Swift behaviour. The driving models
checked so far record every shape and type these passes read.
