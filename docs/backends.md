# Backends and performance

A backend prepares and runs the model on your hardware. Keep the default for
normal use: TensorRT on NVIDIA, or ONNX Runtime with CoreML on Apple silicon.
For installation, see [platform setup](platforms.md).

Use this page to compare runtimes and check their requirements. Detailed Mac
measurements are in the [performance reference](mac-performance.md).

## Runtime comparison

| Backend | Devices | Prepared files | Requirements |
| --- | --- | --- | --- |
| `trt` | NVIDIA CUDA | `.plan` | TensorRT 10.3 on JetPack 6, 10.16 on JetPack 7.2, or 11.x from PyPI on a PC |
| `ort` | CoreML, CUDA, CPU | `.ortcache/` | ONNX Runtime 1.22+ |

`--backend auto` selects TensorRT if available, then ONNX Runtime: CoreML on
macOS, CUDA or the CPU elsewhere. Prepared files are cached separately for each
runtime version and device.

On a Mac, `--device` picks the CoreML layout: `ane` (the default on Apple
silicon) runs the vision trunk on the Neural Engine and the rest on the GPU,
`coreml` runs everything on the GPU, and `ane-whole` runs the whole model as
one CoreML program with every compute unit allowed, prepared the way the
iPhone prepares it (the policy's LayerNormalizations fed inputs scaled by 1/8
so their fp16 squares do not overflow, and the small heads after the trunk in
fp32). `ane-whole` is for comparing against the default, not for daily use;
see [how the default runs](mac-performance.md#how-the-default-runs) for why.

## Platform matrix

| Platform | Backend | USB | Telemetry | Sleep support |
| --- | --- | --- | --- | --- |
| Jetson Orin | TensorRT | USB-A host with libusb | Tegra sensors | Suspend and poweroff |
| Linux with NVIDIA GPU | TensorRT; ONNX Runtime as an alternative | libusb with `scripts/99-jetlink-host.rules` | NVML | `--sleep-after` requires `/sys/power`; USB wake depends on hardware |
| Windows with NVIDIA GPU | TensorRT in WSL2 | Requires `usbipd-win` | NVML | None |
| macOS with Apple silicon | ONNX Runtime with CoreML on the Neural Engine and GPU | USB-A hub, dock, or adapter with libusb | Not available | `scripts/run-mac.sh` prevents idle sleep on AC power |

See [performance and operating limits](status.md) for timing and power considerations.

<a id="how-the-default-runs"></a>
<a id="how-to-measure"></a>
<a id="keeping-the-mac-gpu-responsive-between-frames"></a>
<a id="model-preparation"></a>

## Mac, measured

On a 16 GB M1 Pro, the default Neural Engine/GPU backend averaged about 31 ms
per frame in paced 20 Hz tests, and GPU-only 41 to 44 ms.

These are bench measurements. Some CoreML runs still had individual frames
over the deadline; averages alone do not establish driving reliability.
See the [full measurements and test conditions](mac-performance.md).

| If you need to... | Read |
| --- | --- |
| Compare latency, preparation time, and disk use | [Measured results](mac-performance.md) |
| Understand the Neural Engine/GPU split | [How the default runs](mac-performance.md#how-the-default-runs) |
| Reproduce the tests | [How to measure](mac-performance.md#how-to-measure) |
| Investigate intermittent GPU latency | [GPU keep-alive](mac-performance.md#keeping-the-mac-gpu-responsive-between-frames) |
| Understand CoreML engine preparation | [Model preparation](mac-performance.md#model-preparation) |

## Runtime implementation and dependencies

- ONNX Runtime sessions run in a worker process so model preparation does not
  block server connections, progress updates, or pings. Frame inputs and
  outputs use shared memory.
- Jetlink disables ONNX Runtime telemetry to avoid a macOS shutdown crash.

## Hardware limitations

Native Windows USB requires WinUSB. Use WSL2 for the
[Windows setup](platforms.md#windows-nvidia-gpu). On a Mac, use a USB-A port on a hub, dock, or adapter to ensure
that the Mac acts as the USB host.

Keep laptops powered and awake. Sustained GPU use can cause thermal throttling;
check frame times during use. NVIDIA systems can report GPU telemetry through
NVML with `pip install "jetlink[nvml]"`.
