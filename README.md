<p align="center">
  <img src="macos/Resources/Assets.xcassets/AppIcon.appiconset/icon_128x128@2x.png" width="128" height="128" alt="Jetlink icon">
</p>

<h1 align="center">Jetlink</h1>

<p align="center">
  Run openpilot's large driving models on a computer connected to your comma.
</p>

**Jetlink is experimental.** It needs zoompilot's
[`jetson-trt` branch](https://github.com/zoompilot/zoompilot/tree/jetson-trt).
If the link drops while engaged, the comma soft-disables: take over.
Read the [operating limits](docs/status.md) first.

## Quick start

You need:

* A **comma 3X or comma 4**.
* A **USB 3 USB-C cable** (Jetson: USB-A to USB-C).
* **Separate power for the comma and computer**.

Set up in or out of the car, offroad. Keep the comma online and the computer
powered and awake.

1. Install Jetlink on your computer (below).
2. Do [comma setup](#comma-setup-all-platforms).
3. Wait for the comma's icon to turn green.

### Jetson

Follow the **[Jetson setup guide](docs/jetson.md)** (power, JetPack, installer).

<details>
<summary>Watch the installer</summary>

<a href="docs/images/install-demo.mp4"><img src="docs/images/install-demo.webp" width="100%" alt="Jetlink installer on a Jetson"></a>

</details>

<a id="jetson-or-linux-pc"></a>

### Linux PC

Needs Ubuntu or Debian and an NVIDIA GeForce RTX 20 series or newer GPU. Run:

```bash
curl -fsSL https://raw.githubusercontent.com/zoompilot/jetlink/main/install.sh | bash
```

Installs the latest release; `jetlink update` moves to the newest. Takes 10–30
minutes. Help: [Linux setup](docs/platforms.md#linux-nvidia-gpu).

### Mac

<a href="docs/images/mac-demo.mp4"><img src="docs/images/mac-demo.webp" width="100%" alt="Jetlink for Mac: Use Model downloads and prepares a model, the comma connects over USB, and Status shows each frame against the 50 ms budget"></a>

Needs Apple silicon and macOS 15 or later (16 GB memory recommended).

1. Download the Mac DMG from [Releases](https://github.com/zoompilot/jetlink/releases).
2. Drag **Jetlink** to **Applications** and open it.
3. Wait for **Waiting for comma**.
4. Keep the Mac powered and awake.

Nothing else to install: the server is built into the app.

### iPhone and iPad (experimental)

<a href="docs/images/iphone-demo.mp4"><img src="docs/images/iphone-demo.webp" width="100%" alt="Jetlink for iPhone, recorded in the iOS Simulator: Get downloads and prepares a model, the comma connects over USB, and Status shows each frame against the 50 ms budget, with timings modeled on an iPhone 17 Pro measurement"></a>

Only tested with a Mac standing in for the phone. You need:

* An iPhone or iPad with USB-C on iOS or iPadOS 26.1 or later. For USB 3: an
  iPhone 15 Pro or later Pro, or an iPad Pro, Air, or mini.
* A USB 3 USB-C cable.
* A Mac with Xcode 26.

1. Build and install with Xcode:
   **[Jetlink for iPhone and iPad](docs/iphone-app.md)**. A free Apple account
   works, but you must run it from Xcode again every 7 days.
2. Open Jetlink and keep it on screen.

### Android (experimental)

<a href="docs/images/android-demo.mp4"><img src="docs/images/android-demo.webp" width="100%" alt="Jetlink for Android, recorded in the Android emulator: Get downloads a model, the comma connects over USB, and Status shows each frame against the 50 ms budget. The download is sped up, the emulator's CPU preparation cut, and the model's time modeled on an estimate for a Snapdragon 8 Gen 3"></a>

Not yet run on a phone. You need:

* An Android phone with a Snapdragon 8 Gen 2 or newer and USB 3.
* A USB 3 hub with USB-C power pass-through, and a USB-A to USB-C cable.
* A Mac or Linux PC to build the app.

1. Build and install it: **[Jetlink for Android](docs/android-app.md)**.
2. Open Jetlink, plug in the comma, and tick **Always open** when Android asks.

## Comma setup (all platforms)

1. **Install zoompilot with Jetlink.** After resetting the comma, enter
   **`zoompilot/jetson-trt`** as the install URL (works from any fork).
   Already on zoompilot? Select **jetson-trt** in
   **Settings > Software > Target Branch > Non-Prebuilt Branches**.
   Let it finish installing, rebooting, and building.
2. **Enable Jetlink.** Set **Settings > Models > Accelerator Link** to **USB**
   (Jetson, Linux PC, Mac, Android) or **iOS** (iPhone, iPad). Offroad only. Leave
   **Big Model** at its default for the first run.
3. **Connect USB.** A USB 3 USB-C cable, or a USB-A to USB-C cable with a
   USB-C adapter. Jetson: use its USB-A port. Charge-only cables won't work.
4. **Wait for green.** The home-button icon pulses while the model downloads
   and prepares (first time: about 3 minutes on a Jetson, 20 seconds on an
   M1 Pro). Stay offroad and online until it turns green.

## What to expect when driving

* The small model drives until the large model is ready.
* The large model takes over **at a stop with cruise off, or with lateral
  control off**. With lateral control always on, disengaging alone is not enough.
* Dimmed green icon: waiting to switch. **Big Model Ready**: switched.
* **Big Model Lost** while engaged: take over. The comma soft-disables and
  falls back to the small model.

More: [daily use and icon meanings](docs/using-jetlink.md).

## If something is wrong

| Problem | First check |
| --- | --- |
| No Accelerator Link setting | Confirm the `jetson-trt` branch in Settings > Software. |
| Server stays waiting; icon never pulses | Check the server is running. Try another USB 3 data cable, or a USB-A to USB-C cable with a USB-C adapter. |
| Model list is empty | Connect the comma to the internet, then use Refresh Model List. |
| Setup alert or orange icon | Read the alert. Check internet, then set Accelerator Link to Off and back to USB or iOS. |
| Alert says **no warp built for this camera** | Update or reinstall the `jetson-trt` branch. |
| Link drops repeatedly | Check the cable, separate power, cooling, and computer sleep. |

Logs and more: [Jetson](docs/jetson.md#troubleshooting),
[Mac](docs/macos-app.md#troubleshooting),
[iPhone and iPad](docs/iphone-app.md#logs),
[Android](docs/android-app.md#logs),
[PC](docs/platforms.md#troubleshooting).

<a id="more"></a>

## Documentation

[All guides and references](docs/README.md) ·
[Daily use](docs/using-jetlink.md) ·
[Updates and rollback](docs/releasing.md) ·
[Performance and limits](docs/status.md)

## License

[MIT](LICENSE).
