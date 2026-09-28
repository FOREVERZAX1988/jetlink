# Mac performance measurements

For backend selection and requirements, see [backends](backends.md).
This page preserves the measured results, test conditions, and implementation
details behind the Mac defaults.

These results use a 16 GB M1 Pro, macOS 26.5, ONNX Runtime 1.29.0, tinygrad
0.14.0 at `e837e367aac9`, and the 766 MB Cinque Terre V3 (`404a18cfd86d2963`)
and V2 (`09d080f36965bb2a`) models. Performance on other Macs may differ.

The frame budget is 50 ms at 20 frames per second (20 Hz). On Apple silicon the
default runs the model's vision layers on the Neural Engine and the rest on the
GPU. `--device coreml` runs everything on the GPU. It is slower, but use it if
another app keeps the Neural Engine busy: the default assumes Jetlink is the
only thing using it. tinygrad exceeds the budget.

| | Default: Neural Engine and GPU | GPU only (`--device coreml`) | tinygrad METAL |
| --- | ---: | ---: | ---: |
| V3 round trip at 20 Hz through the server, mean / p99 / max | 30.6 / 33.8 to 34.8 / 40.1 ms | 43.7 / 44.5 to 45.0 / 52.1 ms | not measured |
| V2 round trip at 20 Hz through the server, mean / p99 / max | 30.7 / 32.1 to 37.2 / 39.7 ms | 41.5 / 41.7 to 50.2 / 73.3 ms | not measured |
| frames over the 50 ms budget | V3 0 of 1,740, V2 0 of 1,160 | V3 1 of 1,160, V2 9 of 1,160 | 390 of 390 |
| parity gate, worst column (V3 / V2) | 0.99957 / 0.99957 pass | V2 0.99957 pass | 0.99954 pass |
| build / load in a fresh process | about 20 s / 0.6 to 11 s | about 10 s / 1.8 to 4.7 s | 13 s / 1.1 s |
| artifact on disk | 2.1 GB | 2.3 GB | 777 MB |

The tinygrad column is an earlier measurement of Cinque Terre at 20 Hz: 66.2 ms
mean, 67.6 ms p99, against 43.3 ms and 44.4 ms for the GPU in the same run. The
default and GPU columns were measured on 2026-09-26 in 300-frame blocks, each
against the code before the change it measures, alternating between the two
servers: six blocks for the default on V3, four for the rest. Another process
was busy on the Mac throughout, and in the last GPU-only V2 block it got busier
(43.6 ms mean, 7 frames over), which is most of that column's p99 range and
misses; the other three blocks ran 40.8 to 41.0 ms. The p99 is the range over
the blocks. A load of the default takes under a second when the same model was
the last one loaded, and 5 to 11 s after another, while macOS prepares the
Neural Engine's part again.

The mean is the average frame time. The p99 is the time at or below which 99% of
frames complete. The maximum is the slowest frame.

## The Python server and the Swift server

The app can run either server (Settings > Server). Both run the same prepared
graph through onnxruntime's CoreML provider, so the difference is everything
around the model: the queues, the copies and the process layout. The Python
server runs the model in a worker process; the Swift server runs it in the
app's own process.

Measured 2026-09-27 on the same M1 Pro with Cinque Terre V3, the default
Neural Engine and GPU split, `bench_link.py --rate 20 --n 1200` over TCP
loopback, one server at a time. The Swift server is `jetlink-serve` built for
release, the Python server `python -m jetlink.server.main --transport tcp`.

| Run | round trip p50 / p99 / max | server-side total | over 50 ms |
| --- | ---: | ---: | ---: |
| Python 1 | 31.20 / 57.63 / 96.35 ms | 30.92 ms | 18 of 1,190 |
| Python 2 | 31.16 / 51.34 / 145.63 ms | 30.50 ms | 15 of 1,190 |
| Python 3 | 31.03 / 34.54 / 45.07 ms | 29.92 ms | 0 of 1,190 |
| Swift 1 | 29.81 / 33.51 / 36.90 ms | 28.87 ms | 0 of 1,190 |
| Swift 2 | 30.14 / 33.90 / 34.76 ms | 29.06 ms | 0 of 1,190 |

Another process was building in a container during the first two Python
runs; the third ran under the same load as the Swift runs. On the clean runs
the Swift server is about 1 ms faster a frame at p50 and 0.6 to 1 ms at p99.
That is the server alone: loopback TCP adds about 1.4 ms either way.

Over USB with a comma, only the Python server has been measured so far. On
2026-09-27, with a comma four, the Jetlink v0.4.3 app (the Python server,
Neural Engine) on this M1 Pro and a live bench while parked (big model frame
times, as the comma sees them):

| Cable | p50 | p99 | Dropped |
| --- | ---: | ---: | ---: |
| USB 3 C-to-C | 36.9 ms | 45.7 ms | 0 |
| USB 2 C-to-C | 46.7 ms | 54.3 ms | 0.88% |

Not yet measured: the Swift server over USB, which is the gate for
making the Swift server the default (its p99 no worse than the Python
server's, and no frame dropped). To run it, plug the comma into the Mac, set
Settings > Server to each in turn (or run `jetlink-serve --usb` from
`JetlinkKit/.build/release`), and on the parked comma run
`jetlink_repo/scripts/comma/jetlink_live_bench.sh 180` for each.

One cache serves both servers. The two write the same artifact under the same
name, and once their prepare versions agree ([conformance](conformance.md))
each loads what the other built: the Python server loaded an engine the Swift
server had prepared, with the converted model the Swift server deletes after
compiling already gone, and served it at 29.7 ms p50 with no frame over 50 ms.

## How the default runs

The default (`--device ane`, which `auto` picks on Apple silicon) runs the
convolutional trunk, which reads the camera frames, on the Neural Engine in
about 20 ms, where the GPU takes 31 ms, and everything after it on the GPU.
Jetlink cuts the model where the trunk ends and runs it as two CoreML
sessions, which exchange 32 KB per frame. It does this for every model: V3's
policy and history, and V2's policy with the history the server keeps. V3's
history stays in the worker process that runs the sessions, fed from each
frame's outputs to the next frame's inputs there, rather than crossing to the
server and back as 12 MB a frame (worth 0.6 ms mean and 1.1 ms p99).

The alternative is one session that lets CoreML choose among all compute
units. Mean / p99 in ms at 20 Hz, interleaved on 2026-09-25:

| | V3 | V2 |
| --- | ---: | ---: |
| two sessions, trunk on the Neural Engine | **32.2 / 36.6** | 29.7 / 33.5 |
| one session, every compute unit | 114.8 / 123.2 | 28.6 / 31.5 |
| GPU only | 43.1 / 44.7 | 43.6 / 46.2 |

One session is unusable on V3, because the Neural Engine cannot run its
stateful policy efficiently. On V2 it was about 1 ms faster, but only with the
policy's LayerNormalizations forced into fp32 to keep them off the Neural
Engine and one CPU core kept spinning for CoreML's work in each frame. Jetlink uses the two-session layout for both models to support V3 consistently.
That one-session layout is available as `--device ane-whole`, prepared the way
the iPhone prepares it (the policy's LayerNormalization inputs scaled by 1/8
in fp16, the heads after the trunk in fp32), for A/B runs against the default.

The cut also keeps the Neural Engine's fp16 LayerNormalization out of the
layers after the trunk: with them on the Neural Engine, `road_transform` fell
to a correlation of 0.9988 over 32 frames and failed the parity gate.

Every CoreML build, the GPU-only one too, rewrites two Expand operations
CoreML will not take as the equivalent Tiles, so the policy stays one CoreML
program instead of two with a CPU step between them (worth 3.7 ms mean and 9 ms
p99 on the default with V3, and with FastPrediction 3.8 ms mean on the GPU
alone with V2), and asks CoreML for its FastPrediction specialization. The
default runs the Metal keep-alive described below for its GPU half (without it
the split measured 46.4 ms mean and 53.5 ms p99).

Other apps can use the Neural Engine too, and the default slows down when they
do: with another process running a model on it back to back, the split measured
52.5 ms mean and 65 ms p99 where the GPU option measured 43.5 ms. Switch to
`--device coreml`, or **CoreML on the GPU** in the Mac app, if that happens.

## How to measure

`scripts/verify_parity.py` compares 32 frames against ONNX Runtime on the CPU,
using the model's hidden-state feedback. Every output slice and column must have
a correlation of at least 0.999 to pass. `scripts/verify_engine.py` checks a
prepared engine on the machine that built it, without the link; with
`--capture` it replays a `verify_parity.py` capture and must match what the
comma received, bit for bit.

`scripts/bench_link.py --rate 20` measures round-trip latency through the server
over TCP loopback. Use the 20 Hz results when assessing the driving frame
budget. See [test without a comma](platforms.md#test-without-a-comma).
On the comma, `scripts/comma/jetlink_replay.py` replays a recorded segment
through the real modeld on the accelerator.

## Keeping the Mac GPU responsive between frames

CoreML's GPU path enables a small Metal keep-alive workload while inference
requests are arriving. On an M2 Pro, the gaps in a 20 Hz stream allowed GPU
clocks to fall even though continuous inference met the 50 ms deadline. The
machine reported nominal thermal pressure. A similar intermittent GPU workload
problem and a small-workload workaround are described in
[Anukari's development report](https://anukari.com/blog/devlog/apple-performance-progress).

On an M2 Pro with ONNX Runtime 1.29.0 and model `09d080f36965bb2a`, five-minute
TCP loopback runs at 20 Hz on 2026-09-21 measured:

| | Original run | With keep-alive |
| --- | ---: | ---: |
| mean round trip | 44.13 ms | 35.41 ms |
| p99 round trip | 64.66 ms | 38.62 ms |
| maximum round trip | 83.57 ms | 70.20 ms |
| frames exceeding 50 ms | 1,119 / 5,990 (18.68%) | 3 / 5,990 (0.05%) |

Each run excludes ten warm-up frames. With keep-alive, every 30-second
window had a mean below 35.6 ms and p99 below 39 ms. Three isolated deadline
misses remained; this is a desktop TCP measurement, not USB end-to-end validation.
A subsequent 90-second control run with the helper disabled missed 498 of
1,790 deadlines (27.82%), with a 69.10 ms p99. The first 32 recurrent frames
produced bit-for-bit identical outputs with the helper enabled and disabled.

The helper uses a separate 128-byte buffer, with one finite command in flight
at a time on its own thread. It does not change model inputs, hidden state,
precision, or CoreML compute units. It stops submitting work after one second
without an inference request, on inference errors, or when the worker exits.
It runs whenever a session uses the GPU, including the GPU half of the default;
CPU sessions do not start it. If Metal initialization or a
helper command fails, inference continues without the helper and logs a warning.

This trades additional GPU activity and power consumption for lower latency;
it does not change thermal limits or force a GPU clock setting. To disable it
for comparison, launch the server with `JETLINK_METAL_KEEPALIVE=0`:

```bash
JETLINK_METAL_KEEPALIVE=0 JETLINK_TRANSPORT=tcp \
  scripts/run-mac.sh --backend ort --device coreml --host 127.0.0.1
```

Use the same model and a sustained paced benchmark (`--rate 20 --n 6000`).
A continuous benchmark (`--rate 0`) alone does not establish that the server
can meet deadlines with pauses between frames. Measure on the target machine;
results depend on hardware and competing workloads.

## Model preparation

CoreML stores weights in a binary file. A prepared GPU model uses about 2.3 GB,
takes about 8 seconds to build, and loads in about 2 seconds on the M1 Pro.

For the default, Jetlink also normalizes negative Gather indices, which the
Neural Engine mishandles, and splits the model after the trunk, as above. It
takes about 20 seconds to build. A CoreML engine prepared by an earlier version
of Jetlink is rebuilt automatically.
