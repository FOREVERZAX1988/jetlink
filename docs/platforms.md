# Platform setup

Run Jetlink on a Mac, Linux PC, or Windows PC with WSL2. Set up the comma with the steps
in the [README](../README.md#quick-start).

You can set up Jetlink outside the car or in the car while offroad. Keep the
comma and computer connected to the internet during setup, and keep the
computer powered and awake.

Choose your setup:

- [Mac app](macos-app.md) for installation without terminal commands.
- [Linux installer](#linux-nvidia-gpu) for an NVIDIA PC.
- [Windows WSL2](#windows-nvidia-gpu) for a Windows PC with an NVIDIA GPU.
- [Docker](#docker-nvidia-laptops-and-desktops), [CPU](#cpu-only), or
  [test without a comma](#test-without-a-comma) for development.

For a Jetson, use the [Jetson guide](jetson.md).

## Before running from source

The installer and Mac app do not need a checkout. For the terminal and Docker
examples below, clone the repository first:

```bash
git clone https://github.com/zoompilot/jetlink.git
cd jetlink
```

## Mac (Apple silicon)

Use the Mac app for setup without terminal commands. Download it from
[Releases](https://github.com/zoompilot/jetlink/releases), open it, and leave it
running. It includes Python and does not require Homebrew. The [Mac
guide](macos-app.md) covers installing it, preparing a model ahead of a drive,
and its settings.

### From a terminal

Install Python 3.10 or later and libusb with Homebrew:

```bash
brew install python libusb
scripts/run-mac.sh
```

The first run installs dependencies and starts the server. Plug the comma in
with a **USB 3 USB-C cable**, or a USB-A to USB-C cable with a USB-C adapter.

With the comma's **Accelerator Link** set to USB, the comma presents only the
Jetlink link: no network interface appears on the Mac, a Jetson or a Linux PC.
The network interface an iPhone needs appears only when it is set to iOS; see
[what the comma presents](transport.md#what-the-comma-presents).

Runtime and storage:

- CoreML takes **about 20 seconds** to prepare the model the first time, and
  from under a second to about 10 seconds to load it again every time the server restarts.
- The script holds the Mac awake on AC power. On battery, keep the lid open.
- Models and prepared engines live in `models_cache/` in the checkout,
  about 3 GB per model with CoreML: a 766 MB download plus a 2.1 GB engine.
  Set `JETLINK_CACHE` to move them.

Options:

```bash
# Serve a test client over TCP instead of the comma (see Test without a comma)
JETLINK_TRANSPORT=tcp scripts/run-mac.sh

# The GPU only, if another app keeps the Neural Engine busy
scripts/run-mac.sh --device coreml

# Prepare a model ahead of time, then exit
scripts/run-mac.sh --build /path/to/big_driving_supercombo.onnx
```

For performance comparisons, see [backends and measurements](backends.md#mac-measured).

## Linux (NVIDIA GPU)

For a PC or laptop with a GeForce RTX 20 series or newer GPU, on Ubuntu or
Debian. Run the installer:

```bash
curl -fsSL https://raw.githubusercontent.com/zoompilot/jetlink/main/install.sh | bash
```

The installer checks for NVIDIA driver 580 or newer. On Ubuntu, it can install
the driver for you; restart and run the installer again if prompted. It then
installs Jetlink and asks whether to start it with the computer.

Run `jetlink status` to check it. See [everyday commands](jetson.md#everyday-use)
for logs, updates, and uninstalling.

Plug the comma into a USB-A port, and keep the computer powered and awake while
driving: sleep drops the link.

### Without Docker

For development. You need a working NVIDIA driver and Python 3.10 or later.

```bash
sudo apt update
sudo apt install -y python3-venv libusb-1.0-0
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[trt,usb,nvml]"
sudo install -m 644 scripts/99-jetlink-host.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
jetlink-server --backend trt --transport usb
```

The udev rule grants USB access without root; unplug and replug the comma after
installing it. In a new terminal, run `source .venv/bin/activate` before using
`jetlink-server` again.

Optionally use `jetlink-models` to download and prepare a model before
connecting the comma. Stop the server before preparing into its cache. See
[model management](models.md).

## Windows (NVIDIA GPU)

Use **Ubuntu in WSL2** and follow the Linux steps inside it, or use
[Docker](#docker-nvidia-laptops-and-desktops). Start with a [TCP
test](#test-without-a-comma). USB from WSL2 needs `usbipd-win` to attach the
comma to Ubuntu.

## Docker (NVIDIA laptops and desktops)

The installer uses these images; this section is for running them yourself,
for example on Windows with WSL2. The image includes Python, CUDA 13, TensorRT,
and USB support. The host needs the NVIDIA driver, 580 or newer.

| Image | For |
| --- | --- |
| `ghcr.io/zoompilot/jetlink:VERSION-cuda` | NVIDIA PCs (x86-64) and Jetsons on JetPack 7.2 or newer: one tag, and Docker pulls the right architecture |
| `ghcr.io/zoompilot/jetlink:VERSION-jetpack6` | Jetsons on JetPack 6 (also tagged `-jetson`) |
| `ghcr.io/zoompilot/jetlink:edge-cuda`, `edge-jetpack6` | the newest `main`, what the installer uses |

**Enable GPU access.** On Linux, install Docker Engine and the [NVIDIA Container
Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
then:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

On Windows, install Docker Desktop with the WSL 2 backend and follow [Docker's
GPU guide](https://docs.docker.com/desktop/features/gpu/). Run the remaining
commands in your Ubuntu WSL terminal. Verify GPU access:

```bash
docker run --rm --gpus all nvidia/cuda:13.2.1-base-ubuntu24.04 nvidia-smi
```

**Pull or build.** Pull a release image, replacing `VERSION` with the release
version, such as `0.4.0`, or use `edge-cuda`:

```bash
docker pull ghcr.io/zoompilot/jetlink:VERSION-cuda
docker tag ghcr.io/zoompilot/jetlink:VERSION-cuda jetlink:cuda
```

To build it yourself instead, from the checkout (`docker/Dockerfile.jetpack6`
on a JetPack 6 Jetson):

```bash
docker build -f docker/Dockerfile -t jetlink:cuda .
```

**Run it.**

```bash
docker volume create jetlink-cache
docker run --rm -it --gpus all --name jetlink-cuda \
  -p 127.0.0.1:5599:5599 \
  -v jetlink-cache:/var/cache/jetlink \
  jetlink:cuda
```

This serves TCP on port 5599 for a [test](#test-without-a-comma). For USB on
native Linux, run instead:

```bash
docker run --rm -it --gpus all --name jetlink-cuda \
  --device-cgroup-rule 'c 189:* rmw' \
  --mount type=bind,source=/dev/bus/usb,target=/dev/bus/usb \
  -v jetlink-cache:/var/cache/jetlink \
  jetlink:cuda --transport usb
```

Logs: `docker logs -f jetlink-cuda`. If the container name is in use, run
`docker stop jetlink-cuda` first. Laptop sleep disconnects the link. Keep the
laptop powered and awake.

## CPU only

For checking the protocol and model loading without a GPU. It will not keep up
with driving.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[ort]"
jetlink-server --backend ort --device cpu --transport tcp
```

## Test without a comma

You need a large driving-model ONNX file. With a TCP server running (on an
installed Jetson or PC: `jetlink stop`, then `sudo docker/run.sh --transport
tcp` from a checkout), run these commands from the
checkout in a second terminal. Replace `/path/to/big_model.onnx` with your model
file path:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e . onnx
python3 scripts/bench_link.py --host 127.0.0.1 --onnx /path/to/big_model.onnx --rate 20
```

The benchmark uploads the model, waits for it to build, and reports round-trip
latency and how many frames exceeded the 50 ms budget at 20 Hz. For a server on
another machine, use its wired-network IP. TCP has no authentication, so use a
trusted network. Wi-Fi does not meet the frame budget.

With the Docker image, the same test runs without installing Python. Put the
model in a `models` folder inside the checkout and run:

```bash
docker run --rm -it --network container:jetlink-cuda \
  --mount "type=bind,source=$(pwd)/models,target=/models,readonly" \
  --entrypoint python jetlink:cuda \
  scripts/bench_link.py --host 127.0.0.1 --onnx /models/big_model.onnx --rate 20
```

## Troubleshooting

| Problem | Check |
| --- | --- |
| `python3` too old or not found | Install Python 3.10+ and reopen the terminal |
| `jetlink-server` not found | Run `source .venv/bin/activate` from the project folder |
| Backend missing | `jetlink-server --list-backends`; check that platform's dependencies and GPU driver |
| USB library error | Install native libusb as well as the Python package |
| USB permission error on Linux | Install the udev rule, then replug the comma |
| TCP connection refused | Start the server with `--transport tcp`. Check the IP address and allow port 5599 through the firewall. |
| GPU not found in Docker | Redo the GPU access setup and rerun the `nvidia-smi` check; the driver must be 580 or newer |
| Mac looks stuck loading | CoreML prepares in about 20 seconds and loads in up to 10 on an M1 Pro. If loading takes minutes, remove the prepared engine and prepare it again. Check the server output for errors. |
| Link drops when laptop sleeps | Keep it awake, powered, and open |

Desktop caches use `JETLINK_CACHE` if set, otherwise `~/.cache/jetlink`, or
`~/Library/Caches/jetlink` on a Mac. The Mac script sets it to `models_cache/`.
For comma-side alerts, see the [README](../README.md#if-something-is-wrong).
