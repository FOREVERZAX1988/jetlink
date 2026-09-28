# Jetlink for iPhone and iPad

**Experimental, and not yet run on a phone.** Jetlink for iPhone and iPad runs
the Jetlink server inside an iOS app and serves the comma over one USB cable.
Everything in it has been tested on a Mac, where the same Swift server runs
Cinque Terre V3 at 30 ms per frame. How fast an iPhone runs the model, and
whether it keeps up in a warm car, is still unmeasured; the app's Benchmark
tab is how you find out before you drive.

The same app runs on an iPad with USB-C, and everything here applies to one:
where this page says iPhone or phone, read iPad.

For a Mac, see [Jetlink for Mac](macos-app.md). For a Jetson or PC, see the
[README](../README.md#quick-start).

## How it differs from the Mac app

| | Mac | iPhone and iPad |
| --- | --- | --- |
| Server | The Python server, in an embedded runtime | A Swift port of the server core, in the app |
| Runtime | onnxruntime 1.29 with CoreML | The same: onnxruntime 1.29 with CoreML, the same options |
| Model layout | Vision on the Neural Engine, the rest on the GPU | The whole model on the Neural Engine, whose GPU is far weaker than its Neural Engine. The Mac's split is a setting |
| Link to the comma | USB (the comma is a USB gadget) | USB too, as a network: the comma is a USB network adapter to the phone, and the phone dials it. iOS gives apps no raw USB access |
| While driving | Runs in the background | Must stay on screen. iOS suspends a background app |

## Requirements

| What you need | Why |
| --- | --- |
| An iPhone with USB-C on iOS 26.1 or later, ideally a Pro model from the iPhone 15 Pro on | The cable goes into it. Only those Pro models have a USB 3 port; the other USB-C iPhones are USB 2, which leaves less room in the frame budget |
| Or an iPad with USB-C on iPadOS 26.1 or later, ideally an iPad Pro, iPad Air or iPad mini | Every iPad Pro, iPad Air and iPad mini with USB-C is USB 3 or faster; the iPad (10th generation) and iPad (A16) are USB 2. An iPad with a Lightning port is not supported |
| A USB 3 USB-C cable, or a USB-A to USB-C cable with a USB-C adapter | A powered USB-C hub between them keeps the phone charging: it runs the model 20 times a second |
| About 3 GB of free space per model | A 766 MB download plus the prepared CoreML engine |
| A Mac with Xcode 26 and the iOS 26 platform | There is no App Store or TestFlight build; you build and install it yourself. A free Apple account is enough |

You also need a comma running a zoompilot build with Jetlink, set up as in the
[README](../README.md#comma-setup-all-platforms), with **Accelerator Link** set
to **iOS**.

## Install

There is no App Store build. You sign the app with your own Apple account, and
a free one will do.

1. In Xcode, open **Settings > Components** and install the **iOS 26**
   platform if it is missing. Xcode cannot build for iPhone without it.
2. Under **Settings > Accounts**, add your Apple ID. Without a paid membership
   it appears as a **Personal Team**; the ten-character team ID is beside it.
3. In a checkout, copy `ios/Config/Local.xcconfig.example` to
   `ios/Config/Local.xcconfig` and put your team ID and a bundle identifier of
   your own in it. Git ignores the file. Do not set them under Signing &
   Capabilities, which would write them into the project file.
4. On the iPhone, turn on **Settings > Privacy & Security > Developer Mode**
   and restart. The switch appears once the phone has been plugged into a Mac
   with Xcode open.
5. Install xcodegen (`brew install xcodegen`) and run `make -C ios open`.
   Connect the iPhone, select it as the run destination and click **Run**. The
   **Jetlink** scheme builds Release, which is how the app is used in the car.
   The first build fetches onnxruntime.
6. The first launch is refused until you trust the developer: on the iPhone,
   **Settings > General > VPN & Device Management**, your Apple ID, **Trust**.

The app asks for the Increased Memory Limit capability, as headroom for
preparing a model; a free team can sign it.

With a free account the install stops opening after **7 days**. Click **Run**
again to renew it; the models and settings on the phone are kept. A paid
membership's install lasts a year.

For building and testing without a phone, see [iPhone development](../ios/README.md).

## Connect the comma

One cable. Set to iOS, the comma presents a USB network adapter, which iOS
drives itself. The comma is `192.168.60.1` on that network and gives the phone
an address by DHCP, and the app dials the comma the moment it has one. There is
nothing to type. See [What the comma presents](transport.md#what-the-comma-presents).

1. On the comma, while offroad, set **Accelerator Link** to **iOS** in the
   models settings. USB is for a Jetson, a Linux PC or a Mac; the comma rebuilds
   its USB gadget when the setting moves between the two.
2. Open Jetlink. The first time, allow **Local Network** access when iOS asks.
3. Connect the iPhone or iPad to the comma with a USB 3 USB-C cable, or a USB-A
   to USB-C cable with a USB-C adapter. A powered USB-C hub between them keeps
   it charging.
4. The title reads **Connected over USB 3** once the comma is on. **USB 2**
   there, and on the Link tile, means the phone, the hub or the cable is not
   USB 3; the title turns orange. See [USB 3 matters](#usb-3-matters).

Open the app before plugging in. It dials while the cable is in, so an app
opened afterwards connects when it opens; it just connects later.

The comma holds its USB-C port as the device, so the phone takes the host role
on a direct cable. A direct cable has not been tried with a phone yet: if the
comma restarts when you plug the phone in, connect through a powered USB-C hub.

### USB 3 matters

Every hop has to be USB 3: the phone, the cable and any hub. Apple lists USB 3
only for the Pro models from the iPhone 15 Pro on; the other USB-C iPhones run
at USB 2, and the cable in the box is a USB 2 cable
([Apple](https://support.apple.com/en-us/105099)). Every iPad Pro, iPad Air
and iPad mini with USB-C is USB 3 or faster, and the iPad (10th generation)
and iPad (A16) are USB 2 (the tech specs under
[Identify your iPad model](https://support.apple.com/en-us/108043)).

A frame is about 393 KB up and 74 KB back. At USB 3 the cable costs it about
8.4 ms there and back (p50, 11.2 ms p99), measured with a Mac standing in for
the phone; the comma's USB network driver is the limit, not the wire. USB 2 is
expected to add another 4 to 6 ms; that is an estimate, not yet measured. On
the comma, the negotiated speed is in `/sys/class/udc/*/current_speed`:
`super-speed` is USB 3 and `high-speed` is USB 2.
`sudo scripts/comma/jetlink-root.sh check` prints it.

The app's Settings has these steps under **Help > Connecting the Comma**.

## Prepare a model before you drive

Open **Models** and tap **Get** on the model your comma uses. Jetlink downloads
it, prepares it for this iPhone and loads it, in one step. The download needs
Wi-Fi or cellular data. Keep Jetlink open until it finishes, because iOS
suspends background apps and their downloads.

If you skip this, the comma sends its model when it connects. It drives on its
small model until the phone has the model ready.

A model file can also be copied into the app from the Finder or the Files app,
or imported with **Add Model File**.

## Benchmark

Before the first drive, and after a new model or a change of Processor, find
out whether the phone is fast enough. Open **Benchmark** with a model loaded and
the comma not connected, leave the phone as it will be in the car (charging,
in its mount), and tap **1 Minute**. The app runs the model 20 times a second
on made-up camera frames and times the phone's share of each frame: the
history queues, the model and reading the answer back. The cable is not in it.

While it runs you see the frames so far and the running P50 and P99. At the
end:

- **Verdict.** **Fast Enough** (green) is a P99 at or under 35 ms with nothing
  over 50, which leaves room for the cable. **Tight** (orange) is a P99 under
  50 ms. **Too Slow** (red) misses 20 frames a second.
- **Totals.** Frames, frames over the 50 ms budget (and over 35), the phone's
  temperature at the start and the end, and the model alone.
- **Over Time.** The run ten seconds at a time, each with its P99 and the phone's
  temperature as it closed. This is where a phone that slows as it heats
  shows it.

**10 Minutes** is the same run for long enough to heat the phone. Compare the
first and last windows. The share button puts the whole report on the
clipboard or in a note.

Two things the phone cannot measure itself are filled in as commands to copy,
from the loaded model and the phone's addresses:

- **From the comma**, `jetlink_repo/scripts/comma/jetlink_live_bench.sh 180`
  on the comma over SSH, offroad, with **Accelerator Link** on **iOS** and the
  phone connected. It runs the cameras and modeld as a drive does, over the
  phone's link, and reports the frame times the car will see. Frames over
  50 ms should be 0.
- **From a Mac**, `scripts/verify_parity.py ...` on a Mac on the same Wi-Fi,
  which checks that the phone computes what onnxruntime does on a computer:
  the gate the Mac server passes. It should end with OK.

## Status

Mount the phone where you can see it, in either orientation. The screen stays
on while Jetlink is open. The title's subtitle says where things stand:
**Connected over USB 3** (or **USB 2**, or **Wi-Fi**), **Waiting for Comma**,
**Preparing Model**, **Disconnected**, or what is wrong.

- **Headroom** is the 50 ms frame budget as a ring, filled to the slowest 1%
  of frames (P99) over the last 10 seconds. The number inside is the room
  left. Green is **Good**, orange **Tight** (under 10 ms left), red **Over
  Budget**. The comma's own time and the cable come out of the same 50 ms.
- **Latency** is the average frame, split into Input, Model, Other and Send,
  each measured against the budget.
- **History** has a bar for every 5 seconds of the last 2 minutes, as tall as
  that span's slowest frames.
- **Link** has the frame rate (the comma sends 20 a second) and slow frames,
  and says what the link runs over: USB 3 or USB 2 on the cable, Wi-Fi for a
  bench tool.
- **iPhone** (**iPad** on an iPad) has the phone's temperature, battery,
  memory and link. A hot phone slows down, and it shows here before it shows
  in the ring. Memory is what iOS still lets the app use; it turns orange under
  1 GB, where preparing a model may not fit.

If the app leaves the screen while serving, an orange banner at the top says
so when you come back: iOS suspends an app that is not in front.

On its side the phone shows the ring and the latency with nothing else on
screen. An iPad sets the cards in two columns and keeps its tabs, and in a
narrow Split View or Stage Manager window shows them as an upright phone
does. On the other tabs, the state stays in view in a bar at the bottom; tap
it to go back.

## Logs

**Settings > Help > Logs**, or the document button on Status, shows what the
server and the app have logged: connections, model preparation, slow frames,
memory pressure, heat, and the app going to the background. Warnings are
orange and errors red. The share button sends the whole text; **Clear**
empties the view.

## Heat

A phone running a model 20 times a second warms up, and a phone in a mount in
the sun warms faster. When iOS decides it is too hot it slows the chip down,
and frames start missing 50 ms. The 10-minute benchmark shows how your phone
and mount fare; the Temperature tile and the logs say when it throttles on the
road. Keep the phone out of the sun, and out of a thick case.

## Limits

- **Keep Jetlink on screen.** iOS suspends an app that is not in front, or
  whose phone locks. The comma then drives on its small model, and if engaged
  it asks you to take over. Do not use other apps on the phone while driving.
- **Heat.** See above.
- **The comma cannot power the phone off.** A shutdown request is refused, and
  the app tells you the comma asked.
- **One comma at a time.** A new connection replaces the current one.
- **A free account's install expires after 7 days.** Run it again from Xcode.

## Settings

| Setting | What it does |
| --- | --- |
| Link | USB 3 or USB 2 while the comma is connected (Wi-Fi for a bench tool), Connecting while the phone dials it |
| Port | The TCP port the phone listens on, 5599 by default, for `verify_parity.py` from a Mac. Over the cable the phone dials the comma's port instead |
| Wi-Fi | The phone's Wi-Fi address and port, where a Mac's bench tools reach the phone. The cable's address is automatic and not shown |
| Processor | Neural Engine (default): the whole model on it. Neural Engine + GPU: the Mac's layout, the vision trunk on the Neural Engine and the rest on the GPU. GPU: for when another app keeps the Neural Engine busy. Changing it prepares the model again |
| Keep CPU Awake | On by default. A CPU core kept busy between frames while the Neural Engine runs the model, so the next frame is not waiting on a core that went to sleep. It uses some power |
| Keep GPU Awake | A small GPU job between frames so the GPU does not slow down in the gaps. It uses some power |
| Keep Screen On | On by default. With it off, auto-lock suspends Jetlink |
| Help | How to connect the comma, and the logs |
