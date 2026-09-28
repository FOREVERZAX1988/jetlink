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
branch installation and **Accelerator Link** set to **USB**. Then connect the
**Mac's USB-A port to the comma's USB-C port**, using a USB-A port
on a hub or dock, or a USB-C-to-A adapter. A C-to-C cable also connects: on
2026-09-27 an M1 Pro enumerated a comma four as its device at 5 Gb/s over a
USB 3 C-to-C cable. The cable must be a USB 3 one; C-to-C cables marked for
USB 2 run at 480 Mb/s, about 10 ms slower a frame. If a C-to-C cable ever gives
the comma the host role instead, the zoompilot branch's port hold makes it the
device within a few seconds.

The Status screen then shows:

- **Connected over USB 3** on the Link row. It names the link: **USB 3**,
  **USB 2**, or **TCP** with the client's address. **USB 2** is shown as a
  warning: a frame takes about 10 ms longer to cross, which on the bench cost
  close to 1% of frames, near where the comma soft-disables. Use a USB 3 cable
  and a USB 3 port. The menu bar says the same.
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
| Server | **Python (bundled runtime)**, the default, or **Swift (built in)**, the server the iPhone app runs, inside the app with no Python. See [The Swift server](#the-swift-server). |
| Backend | Which runtime prepares and runs the model. See [Backends](#backends). |
| Connection | **USB (the comma)** for driving, or **TCP** for testing without a comma. |
| Port | The TCP port, 5599 by default. Only shown for TCP. |
| Log level | **Normal (INFO)** or **Verbose (DEBUG)**. Use verbose when reporting a problem. |
| Python interpreter override | For development only. Leave it empty to use the bundled runtime. Shown for the Python server only. |

Click **Restart Server** to apply these settings.

### The Swift server

**Settings > Server > Swift (built in)** runs the server the iPhone app runs,
inside Jetlink: nothing to find or start, the same models and cache folder,
and the same Status, Models and Logs. It opens the comma's USB link itself
through macOS's USB framework, or listens on the TCP port for a bench client.
On an M1 Pro it is about 1 ms faster a frame than the Python server
([measurements](mac-performance.md#the-python-server-and-the-swift-server)).

- It has the same two backends as the Python server, **Automatic** and
  **CoreML on the GPU**.
- The **Benchmark** page (Command-3) runs the loaded model at the comma's pace
  on the Mac alone and gives the same verdict as the iPhone app. It works with
  the Swift server only, and only while no comma is connected.
- It has not yet driven with a comma on USB. Until that is measured, the Python
  server stays the default.

## Backends

Leave **Backend** set to **Automatic**. If another app is using the Neural
Engine and Jetlink slows down, try **CoreML on the GPU**. These comparisons
were measured on an M1 Pro; other Macs may differ.

| Backend | On an M1 Pro | Pick it when |
| --- | --- | --- |
| Automatic (recommended) | The fastest | Use this by default. |
| CoreML on the GPU | About a third slower | Another app keeps the Neural Engine busy. |

If you previously chose **CoreML on the GPU**, select **Automatic** to switch
back. See [backend measurements](backends.md#mac-measured) for details.

## Troubleshooting

| Problem | What to do |
| --- | --- |
| The server failed to start | Open **Logs**. The last lines say why. The usual causes are another server already holding the USB device, and a cache folder that is not writable. |
| The app stays on Waiting for comma | Use a USB-A port on a hub, dock or adapter, use a USB 3 data cable, and check that **Accelerator Link** is set to **USB** under Settings > Models on the comma. |
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

Building the app, the embedded Python runtime, signing and notarizing are
covered in the [Mac developer guide](../macos/README.md).

For scripting, see the [model CLI](model-cli.md) and [control protocol](control-protocol.md).
