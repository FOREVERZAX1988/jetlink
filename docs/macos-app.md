# Jetlink for Mac

Jetlink for Mac runs the server without terminal commands. For the Jetson, see
the [Jetson guide](jetson.md). For the command line on any platform, see
[platform setup](platforms.md).

<a id="what-it-does"></a>

## Requirements

- An Apple silicon Mac with macOS 15 or later. Intel Macs are not supported.
- 16 GB of memory recommended and about 3 GB of disk space per model.
- A USB-A port on a hub, dock, or adapter, and a USB 3 A-to-C data cable.

You also need a comma running a zoompilot build with Jetlink in it, set up with
the steps in the [README](../README.md#quick-start).

## Install

1. Download the Mac DMG from [Releases](https://github.com/zoompilot/jetlink/releases).
2. Open it and drag Jetlink to Applications.
3. Open Jetlink from Applications.

On first launch the server starts by itself and the Status screen says **Waiting
for comma**. Connect the comma to continue.

## Plug in

Complete [comma setup](../README.md#comma-setup-all-platforms), including the
branch installation and **Accelerator Link** toggle. Then connect the
**Mac's USB-A port to the comma's USB-C port**, using a USB-A port
on a hub or dock, or a USB-C-to-A adapter. A plain C-to-C cable may not give the
Mac the host role. The zoompilot `iphone` branch makes the comma hold its port
as the device for a C-to-C host, which should fix that; it has not been tried
with a Mac yet.

The Status screen then shows:

- **Connected over USB** on the Link row.
- **Rate**, the frames per second the comma is sending. It should settle near
  20 per second.
- **Slow frames**, the number of frames over 60 ms in the last second. This should stay at zero. A consistently higher count means the Mac is too slow, and the comma may drop back to its small model.
- **Frame budget**, the time left to finish each frame. Aim for at least
  10 ms to spare. See [performance measurements](mac-performance.md) for details.

On the comma, the home-button icon pulses while the model transfers and loads,
then turns green. For driving behavior, see the
[daily use guide](using-jetlink.md).

The Mac also gains a network service named **jetlink** in System Settings >
Network, with a `192.168.60.x` address and no router. That is the comma's
network interface for an iPhone, part of the same USB gadget as the link; the
Mac app uses the link, not the network, so leave the service alone. See [what
the comma presents](transport.md#what-the-comma-presents).

## Everyday use

Keep the Mac powered and awake. You can close the window; the server keeps
running and the menu bar icon stays. Quitting Jetlink stops the server.
The next launch uses the same model again.

<a id="prepare-a-model-before-you-drive"></a>

## Use a model before you drive

This is optional. The comma can send the model when it connects, but it then
drives on its small model until the Mac has prepared it. Using the model on the
Mac first avoids that wait.

If you have not changed the model on the comma, click **Use** with the default
model's name on the Status screen. One click downloads it, prepares it for this
Mac and starts using it.

For another model, open **Models**. The list is the same one the comma shows
under **Settings > Models > Big Model**, in the same order. Click **Use Model**
in its row, or double-click the row. The row shows the download, then the
preparation, then **In Use**. The cancel button next to a download stops it.
Right-click a model for everything else: **Stop Using Model**, **Show in
Finder**, and deleting its download or its prepared engines. **Inspector**
(Command-I) shows its checksum, files and engines.

Under each model's name are its date, its size, and what is on this Mac:

| Line under the name | What it means |
| --- | --- |
| Date and size only | The model's file is not on this Mac yet. Use Model downloads it first. |
| Downloaded | The file is on this Mac but has not been prepared. Use Model prepares it. |
| Prepared for CoreML | A compiled engine is on disk. Use Model only has to load it. |

Preparing takes about 20 seconds the first time on an M1 Pro, and loading a
prepared engine takes under a second when it was the last model loaded and up to about
10 seconds otherwise.

<details>
<summary>App screenshots</summary>

These screenshots show an earlier app build.

![Server status and model loading](images/mac-status.webp)
![Available models and download status](images/mac-models.webp)

</details>

## Settings

**General**

| Setting | What it does |
| --- | --- |
| Start server when Jetlink opens | Starts the server as soon as the app launches. On by default. |
| Open Jetlink at login | Adds Jetlink as a login item, so it is running before you get in the car. |
| Keep the Mac awake while serving | Prevents idle sleep when connected to power. On battery, keep the lid open. |
| Cache folder | Stores models and prepared engines. A CoreML engine is about 2 GB. Changing it takes effect when the server restarts. |

The cache folder defaults to `~/Library/Application Support/Jetlink/cache`. If
you already used `scripts/run-mac.sh`, you have a `models_cache/` folder in a
checkout. Point the cache folder at it with **Choose…**, and nothing is
downloaded or prepared again.

**Server**

| Setting | What it does |
| --- | --- |
| Backend | Where the model runs. See [Backends](#backends). |
| Connection | **USB (the comma)** for driving, or **TCP** for testing without a comma. |
| Port | The TCP port, 5599 by default. Only shown for TCP. |

Click **Restart Server** to apply these settings.

The server is built into the app: it is the same Swift server the iPhone app
runs, with no Python to install or start. It opens the comma's USB link itself
through macOS's USB framework, or listens on the TCP port for a bench client.

The **Benchmark** page (Command-3) runs the loaded model at the comma's pace on
the Mac alone and gives the same verdict as the iPhone app. It runs only while
no comma is connected.

## Backends

Leave **Backend** set to **Automatic**. If another app is using the Neural
Engine and Jetlink slows down, try **CoreML on the GPU**. These comparisons
were measured on an M1 Pro; other Macs may differ.

| Backend | On an M1 Pro | Pick it when |
| --- | --- | --- |
| Automatic (recommended) | The fastest | Use this by default. |
| CoreML on the GPU | About a third slower | Another app keeps the Neural Engine busy. |

If you previously chose **CoreML on the GPU**, select **Automatic** to switch
back. A tinygrad choice from an earlier version reads as Automatic: tinygrad
went with the Python runtime the app no longer bundles. See [backend
measurements](backends.md#mac-measured) for details.

## Troubleshooting

| Problem | What to do |
| --- | --- |
| The server failed to start | Open **Logs**. The last lines say why. The usual causes are another program already holding the comma's USB interface (a `scripts/run-mac.sh` server, say), and a cache folder that is not writable. |
| The app stays on Waiting for comma | Use a USB-A port on a hub, dock or adapter, use a USB 3 data cable, and check that **Accelerator Link** is on under Settings > Models on the comma. |
| Use Model takes a long time | CoreML should take about 20 seconds to prepare and up to about 10 seconds to load. If it takes minutes, right-click the model in **Models**, choose **Delete Prepared Engines…**, then use it again. Close other large applications to free memory. |
| The comma says **Big Model Lost** | Check the cable first. Then check that the Mac did not sleep: turn on **Keep the Mac awake while serving** and keep the Mac on power. |
| Everything rebuilt after an update | A new runtime version means a new prepared engine, so the model is prepared again. The download is kept and is not fetched twice. |
| The model list is empty | The Mac needs internet for the list. Open **Models** and choose **Refresh**. |
| Frames are slow or the rate is below 20 | Check the cable and the USB port, then check whether another heavy application is using the GPU or the Neural Engine. If one is, choose **CoreML on the GPU** under Settings > Server. |

<details>
<summary>Example server error</summary>

Open **Logs** for the full diagnostic output.

![Server failure and diagnostic output](images/mac-error.webp)

</details>

## Where things live

| What | Where |
| --- | --- |
| Models and prepared engines | `~/Library/Application Support/Jetlink/cache`, or the cache folder you chose |
| Server log | `~/Library/Logs/Jetlink/server.log` |
| The app | `/Applications/Jetlink.app` |

To uninstall, quit Jetlink, then delete `/Applications/Jetlink.app`,
`~/Library/Application Support/Jetlink` and `~/Library/Logs/Jetlink`. If you
turned on **Open Jetlink at login**, remove it in **System Settings > General >
Login Items**.

## For developers

Building, signing and notarizing the app are covered in the [Mac developer
guide](../macos/README.md). The Python server still runs on a Mac from a
checkout, with `scripts/run-mac.sh`, for work on the Python side.

For scripting, see the [model CLI](model-cli.md) and [control protocol](control-protocol.md).
