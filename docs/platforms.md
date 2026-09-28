# Platform setup

Run Jetlink on a Mac, Linux PC, or Windows PC with WSL2. Set up the comma per the
[README](../README.md#quick-start). Jetson: [Jetson guide](jetson.md).

Set up outside the car or in the car while offroad. Keep the comma and computer
online, and the computer powered and awake.

- [Mac app](macos-app.md): no terminal commands.
- [Linux installer](#linux-nvidia-gpu): an NVIDIA PC.
- [Windows WSL2](#windows-nvidia-gpu): a Windows PC with an NVIDIA GPU.
- [Docker](#docker-nvidia-laptops-and-desktops), [CPU](#cpu-only),
  [test without a comma](#test-without-a-comma): development.

## Before running from source

The installer and Mac app need no checkout. For the terminal and Docker steps
below, clone first:

```bash
git clone https://github.com/zoompilot/jetlink.git
cd jetlink
```

## Mac (Apple silicon)

Use the [Mac app](macos-app.md) from
[Releases](https://github.com/zoompilot/jetlink/releases). The server is built
in; no Python or Homebrew needed.

### From a terminal

Needs Python 3.10 or later and libusb from Homebrew:

```bash
brew install python libusb
scripts/run-mac.sh
```

- The first run installs dependencies and starts the server.
- Plug the comma in with a **USB 3 USB-C cable**, or a USB-A to USB-C cable with
  a USB-C adapter.
- Set the comma's **Accelerator Link** to **USB** (also for Android; **iOS** is for iPhone). See
  [what the comma presents](transport.md#what-the-comma-presents).
- CoreML prepares a model in **about 20 seconds** the first time, and loads it
  in under a second to about 10 seconds on every server restart.
- The script keeps the Mac awake on AC power. On battery, keep the lid open.
- Models and prepared engines go in `models_cache/` in the checkout, about 3 GB
  per model. Set `JETLINK_CACHE` to move them.

Options:

```bash
# Serve a test client over TCP instead of the comma (see Test without a comma)
JETLINK_TRANSPORT=tcp scripts/run-mac.sh

# The GPU only, if another app keeps the Neural Engine busy
scripts/run-mac.sh --device coreml

# Prepare a model ahead of time, then exit
scripts/run-mac.sh --build /path/to/big_driving_supercombo.onnx
```

Performance: [backends and measurements](backends.md#mac-measured).

## Linux (NVIDIA GPU)

For a GeForce RTX 20 series or newer GPU, on Ubuntu or Debian:

```bash
curl -fsSL https://raw.githubusercontent.com/zoompilot/jetlink/main/install.sh | bash
```

- The installer needs NVIDIA driver 580 or newer. On Ubuntu it can install it;
  restart and rerun the installer if prompted.
- It asks whether to start Jetlink with the computer.
- Check it with `jetlink status`. Logs, updates, uninstalling:
  [everyday commands](jetson.md#everyday-use).
- Plug the comma into a USB-A port. Keep the computer powered and awake while
  driving: sleep drops the link.

### Without Docker

For development. Needs a working NVIDIA driver and Python 3.10 or later.

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

- The udev rule grants USB access without root. Replug the comma after
  installing it.
- In a new terminal, run `source .venv/bin/activate` before `jetlink-server`.
- To prepare a model before connecting the comma, use `jetlink-models`, with the
  server stopped. See [model management](models.md).

## Windows (NVIDIA GPU)

Use **Ubuntu in WSL2** and follow the Linux steps inside it, or use
[Docker](#docker-nvidia-laptops-and-desktops). Start with a
[TCP test](#test-without-a-comma). USB from WSL2 needs `usbipd-win` to attach
the comma to Ubuntu.

## Docker (NVIDIA laptops and desktops)

The installer uses these images; this is for running them yourself (for
example, on Windows with WSL2). They include Python, CUDA 13, TensorRT and USB
support. The host needs NVIDIA driver 580 or newer.

| Image | For |
| --- | --- |
| `ghcr.io/zoompilot/jetlink:VERSION-cuda` | NVIDIA PCs (x86-64) and Jetsons on JetPack 7.2 or newer: one tag, and Docker pulls the right architecture |
| `ghcr.io/zoompilot/jetlink:VERSION-jetpack6` | Jetsons on JetPack 6 (also tagged `-jetson`) |
| `ghcr.io/zoompilot/jetlink:edge-cuda`, `edge-jetpack6` | the newest `main`, what the installer uses with `--ref main` |

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

**Pull or build.** Pull a release image, replacing `VERSION` with a release
such as `0.4.0`, or use `edge-cuda`:

```bash
docker pull ghcr.io/zoompilot/jetlink:VERSION-cuda
docker tag ghcr.io/zoompilot/jetlink:VERSION-cuda jetlink:cuda
```

Or build it from the checkout (`docker/Dockerfile.jetpack6` on a JetPack 6
Jetson):

```bash
docker build -f docker/Dockerfile -t jetlink:cuda .
```

**Run it.** For a TCP [test](#test-without-a-comma) on port 5599:

```bash
docker volume create jetlink-cache
docker run --rm -it --gpus all --name jetlink-cuda \
  -p 127.0.0.1:5599:5599 \
  -v jetlink-cache:/var/cache/jetlink \
  jetlink:cuda
```

For USB on native Linux:

```bash
docker run --rm -it --gpus all --name jetlink-cuda \
  --device-cgroup-rule 'c 189:* rmw' \
  --mount type=bind,source=/dev/bus/usb,target=/dev/bus/usb \
  -v jetlink-cache:/var/cache/jetlink \
  jetlink:cuda --transport usb
```

- Logs: `docker logs -f jetlink-cuda`.
- If the container name is in use, run `docker stop jetlink-cuda` first.
- Keep a laptop powered and awake: sleep disconnects the link.

## CPU only

Checks the protocol and model loading without a GPU. Too slow for driving.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[ort]"
jetlink-server --backend ort --device cpu --transport tcp
```

## Test without a comma

You need a large driving-model ONNX file and a TCP server running (on an
installed Jetson or PC: `jetlink stop`, then `sudo docker/run.sh --transport
tcp` from a checkout). In a second terminal, from the checkout, replace
`/path/to/big_model.onnx` with your model and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e . onnx
python3 scripts/bench_link.py --host 127.0.0.1 --onnx /path/to/big_model.onnx --rate 20
```

- It uploads the model, waits for the build, and reports round-trip latency and
  frames over the 50 ms budget at 20 Hz.
- For a server on another machine, use its wired-network IP.
- TCP has no authentication: use a trusted network.
- Wi-Fi does not meet the frame budget.

With the Docker image, no Python install is needed. Put the model in a `models`
folder in the checkout and run:

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
| TCP connection refused | Start the server with `--transport tcp`. Check the IP and allow port 5599 through the firewall |
| GPU not found in Docker | Redo the GPU access setup and rerun the `nvidia-smi` check. The driver must be 580 or newer |
| Mac looks stuck loading | On an M1 Pro, CoreML prepares in about 20 seconds and loads in up to 10. If loading takes minutes, remove the prepared engine and prepare again. Check the server output for errors |
| Link drops when the laptop sleeps | Keep it awake, powered, and open |

Desktop caches use `JETLINK_CACHE` if set, otherwise `~/.cache/jetlink`, or
`~/Library/Caches/jetlink` on a Mac (the Mac script sets `models_cache/`).
Comma-side alerts: [README](../README.md#if-something-is-wrong).
