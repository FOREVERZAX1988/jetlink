# Backends and performance

A backend prepares and runs the model. Defaults: TensorRT on NVIDIA, ONNX
Runtime with CoreML on Apple silicon. Install: [platform setup](platforms.md).
Mac detail: [performance reference](mac-performance.md).

The table below is the Python server's, which runs on Jetsons, Linux PCs and
from a checkout on a Mac. The Mac app runs the Swift server instead (the one
the iPhone app runs): ONNX Runtime with CoreML only, in the `ane` and `coreml`
layouts described below, and no tinygrad.

## Runtime comparison

| Backend | Devices | Prepared files | Requirements |
| --- | --- | --- | --- |
| `trt` | NVIDIA CUDA | `.plan` | TensorRT 10.3 on JetPack 6, 10.16 on JetPack 7.2, or 11.x from PyPI on a PC |
| `ort` | CoreML, CUDA, CPU | `.ortcache/` | ONNX Runtime 1.22+ |
| `ort` (Android, Swift server) | QNN: Hexagon NPU, Adreno GPU; CPU | `.ortcache/` | onnxruntime-android-qnn 1.29.0 |

- `--backend auto`: TensorRT if available, else ONNX Runtime (CoreML on macOS,
  CUDA or CPU elsewhere).
- Prepared files are cached per runtime version and device.

On a Mac, `--device` picks the CoreML layout:

| `--device` | Layout |
| --- | --- |
| `ane` (default on Apple silicon) | vision trunk on the Neural Engine, the rest on the GPU |
| `coreml` | everything on the GPU |
| `ane-whole` | one CoreML program, every compute unit allowed, prepared as the iPhone does (policy LayerNormalization inputs scaled by 1/8 so their fp16 squares do not overflow; heads after the trunk in fp32). For comparing against the default, not daily use; see [how the default runs](mac-performance.md#how-the-default-runs). |

The Android app's Swift server has its own devices (Settings > Processor), on
the same preparation as CoreML's:

| Device | Layout |
| --- | --- |
| `htp` (NPU + GPU, default) | vision trunk on the NPU in fp16, the rest on the GPU, as `ane` splits it; the NPU part compiled once into onnxruntime's EP context |
| `htp-whole` (NPU) | the whole graph on the NPU, prepared as `ane-whole`; the NPU computes in fp16 throughout, heads included |
| `gpu` | everything on the Adreno GPU |
| `cpu` | onnxruntime's CPU provider, for the emulator |

None of these has run on a Snapdragon yet.

## Platform matrix

| Platform | Backend | USB | Telemetry | Sleep support |
| --- | --- | --- | --- | --- |
| Jetson Orin | TensorRT | USB-A host with libusb | Tegra sensors | Suspend and poweroff |
| Linux with NVIDIA GPU | TensorRT; ONNX Runtime as an alternative | libusb with `scripts/99-jetlink-host.rules` | NVML | `--sleep-after` requires `/sys/power`; USB wake depends on hardware |
| Windows with NVIDIA GPU | TensorRT in WSL2 | Requires `usbipd-win` | NVML | None |
| macOS with Apple silicon | ONNX Runtime with CoreML on the Neural Engine and GPU | USB 3 USB-C cable, or USB-A to USB-C with a USB-C adapter; the app uses macOS's USB framework, the Python server libusb | Not available | The app, or `scripts/run-mac.sh`, prevents idle sleep on AC power |
| Android with Snapdragon | ONNX Runtime with QNN on the NPU and GPU (the app's Swift server) | USB host through a hub; usbdevfs on the app's descriptor | Not available | A foreground service keeps it serving |

Timing and power: [performance and operating limits](status.md).

<a id="how-the-default-runs"></a>
<a id="how-to-measure"></a>
<a id="keeping-the-mac-gpu-responsive-between-frames"></a>
<a id="model-preparation"></a>

## Mac, measured

16 GB M1 Pro, paced 20 Hz: the default Neural Engine/GPU backend about 31 ms a
frame (about 30 ms in the app's Swift server), GPU-only 41 to 44 ms. Bench
numbers only: some CoreML runs still had single frames over the deadline, so
averages do not establish driving reliability. [Full measurements and test
conditions](mac-performance.md).

| If you need to... | Read |
| --- | --- |
| Compare latency, preparation time, and disk use | [Measured results](mac-performance.md) |
| Understand the Neural Engine/GPU split | [How the default runs](mac-performance.md#how-the-default-runs) |
| Reproduce the tests | [How to measure](mac-performance.md#how-to-measure) |
| Investigate intermittent GPU latency | [GPU keep-alive](mac-performance.md#keeping-the-mac-gpu-responsive-between-frames) |
| Understand CoreML engine preparation | [Model preparation](mac-performance.md#model-preparation) |

## Runtime implementation and dependencies

- ONNX Runtime sessions run in a worker process, so preparation does not block
  connections, progress updates, or pings; frame inputs and outputs use shared
  memory.
- ONNX Runtime telemetry is disabled to avoid a macOS shutdown crash.

## Hardware limitations

- Native Windows is unsupported; use WSL2 ([Windows setup](platforms.md#windows-nvidia-gpu)).
- Mac: a USB 3 USB-C cable, or USB-A to USB-C with a USB-C adapter; the comma
  holds its port as the device, so the Mac is the USB host.
- Keep laptops powered and awake. Sustained GPU use can throttle thermally;
  check frame times.
- NVIDIA GPU telemetry through NVML: `pip install "jetlink[nvml]"`.
