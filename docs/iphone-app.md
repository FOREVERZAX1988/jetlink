# Jetlink for iPhone

**Experimental, and not yet run on a phone.** Jetlink for iPhone runs the
Jetlink server inside an iPhone app and serves the comma over one USB cable.
Everything in it has been tested on a Mac, where the same Swift server runs
Cinque Terre V3 at 30 ms per frame. How fast an iPhone runs the model, and
whether it keeps up in a warm car, is still unmeasured; the app's Benchmark
tab is how you find out before you drive.

For a Mac, see [Jetlink for Mac](macos-app.md). For a Jetson or PC, see the
[README](../README.md#quick-start).

## How it differs from the Mac app

| | Mac | iPhone |
| --- | --- | --- |
| Server | The Python server, in an embedded runtime | A Swift port of the server core, in the app |
| Runtime | onnxruntime 1.29 with CoreML | The same: onnxruntime 1.29 with CoreML, the same options |
| Model layout | Vision on the Neural Engine, the rest on the GPU | The whole model on the Neural Engine, whose GPU is far weaker than its Neural Engine. The Mac's split is a setting |
| Link to the comma | USB (the comma is a USB gadget) | USB too, as a network: the comma is a USB network adapter to the phone, and the phone dials it. iOS gives apps no raw USB access |
| While driving | Runs in the background | Must stay on screen. iOS suspends a background app |

## Requirements

| What you need | Why |
| --- | --- |
| An iPhone with USB-C on iOS 26.1 or later | The cable goes into it. A Pro model's USB 3 port leaves room in the frame budget that USB 2 does not |
| A USB 3 hub with USB-C power passthrough, and a USB 3 A-to-C data cable | The comma plugs into the hub's A port; see [Connect the comma](#connect-the-comma) for why not straight into the phone. The phone runs the model 20 times a second and belongs on power |
| About 3 GB of free space per model | A 766 MB download plus the prepared CoreML engine |
| A Mac with Xcode 26 and the iOS 26 platform | There is no App Store or TestFlight build; you build and install it yourself. A free Apple account is enough |

You also need a comma running a zoompilot build with Jetlink, set up as in the
[README](../README.md#comma-setup-all-platforms).

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
5. Run `make -C ios open`, connect the iPhone, select it as the run destination
   and click **Run**. The **Jetlink** scheme builds Release, which is how the
   app is used in the car. The first build fetches onnxruntime.
6. The first launch is refused until you trust the developer: on the iPhone,
   **Settings > General > VPN & Device Management**, your Apple ID, **Trust**.

The app asks for the Increased Memory Limit capability, as headroom for
preparing a model; a free team can sign it.

With a free account the install stops opening after **7 days**. Click **Run**
again to renew it; the models and settings on the phone are kept. A paid
membership's install lasts a year.

For building and testing without a phone, see [iPhone development](../ios/README.md).

## Connect the comma

One cable. The comma's USB gadget is composite: beside the link a Jetson or a
Mac uses, it presents a USB network adapter, which iOS drives itself. The comma
is `192.168.60.1` on that network and gives the phone an address by DHCP, and
the app dials the comma the moment it has one. There is nothing to type, on
either end. See [What the comma presents](transport.md#what-the-comma-presents).

1. Open Jetlink. The first time, allow **Local Network** access when iOS asks.
2. Plug a USB 3 hub into the iPhone.
3. Join the hub's USB-A port to the comma's USB-C port with a USB 3 A-to-C data
   cable.
4. The title reads **Connected over USB** once the comma is on. Settings shows
   the phone's address on the cable under **Connection**.

Open the app before plugging in. It dials while the cable is in, so an app
opened afterwards connects when it opens; it just connects later.

**Do not plug the comma straight into the phone with a C-to-C cable.** The two
negotiate power, the comma ends up supplying the phone, and it reboots. Through
a hub's A port the comma only ever draws.

A direct cable is being worked on. The zoompilot `iphone` branch has the comma
hold its USB-C port as the device, with USB power delivery off, whenever the
far end of the cable is a host and not a chestnut, so the phone takes the host
role as a hub gives it today. Nobody has tried it with an iPhone yet; use the
hub until the [status page](apple-status.md) says it works.

**USB 3 matters.** A frame is about 460 KB: around 1 ms on USB 3 and around
11 ms on USB 2, which is most of the room in the 50 ms budget. Use a USB 3 hub
and a USB 3 cable. On the comma, the negotiated speed is in
`/sys/class/udc/*/current_speed`: `super-speed` is USB 3 and `high-speed` is
USB 2. `sudo scripts/setup_gadget.sh --check` prints it.

### Ethernet adapter, the manual fallback

If the comma's kernel has no USB network function, the comma dials the phone
over Ethernet instead, as a Jetson on an Ethernet adapter does.

1. Plug a USB-C gigabit Ethernet adapter into the comma and another into the
   iPhone (or a hub with Ethernet), and join them with a network cable. The
   comma supports Realtek RTL8152/8153 and ASIX AX88179 adapters.
2. On the iPhone, open **Settings > Ethernet**, choose the adapter, set
   **Configure IP** to **Manual**, and give it an address such as `10.0.0.2`
   with subnet mask `255.255.255.0`.
3. Give the comma's adapter an address on the same subnet, such as `10.0.0.1`.
4. Set the comma's `JetlinkEndpoint` parameter to the Ethernet address the app
   shows under **Connection**, such as `10.0.0.2:5599`. See
   [Ethernet (TCP)](transport.md#ethernet-tcp).

The title then reads **Connected over Ethernet**. Wi-Fi works for testing,
including the iPhone's Personal Hotspot, but it does not meet the frame budget.

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

Before the first drive, and after a new model or a change of Compute, find out
whether the phone is fast enough. Open **Benchmark** with a model loaded and
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
- **Windows.** The run ten seconds at a time, each with its P99 and the phone's
  temperature as it closed. This is where a phone that slows as it heats
  shows it.

**10 Minutes** is the same run for long enough to heat the phone. Compare the
first and last windows. The share button puts the whole report on the
clipboard or in a note.

Two things the phone cannot measure itself are filled in as commands to copy,
from the loaded model and the phone's addresses:

- **From the comma**, `scripts/bench_link.py --listen ...` on the comma over
  SSH, with the cable in and **Accelerator Link** off on the comma so its own
  client is not holding the port. It waits for the phone to dial, sends 1,200
  real-sized frames and reports the round trip the car will see. Frames over
  50 ms should be 0.
- **From a Mac**, `scripts/verify_parity.py ...` on a Mac on the same Wi-Fi,
  which checks that the phone computes what onnxruntime does on a computer:
  the gate the Mac server passes. It should end with OK.

## Status

Mount the phone where you can see it, in either orientation. The screen stays
on while Jetlink is open. The title's subtitle says where things stand:
**Connected over USB**, **Waiting**, **Preparing**, **Disconnected**, or what
is wrong.

- **Headroom** is the 50 ms frame budget as a ring, filled to the slowest 1%
  of frames (P99) over the last 10 seconds. The number inside is the room
  left. Green is **Good**, orange **Tight** (under 10 ms left), red **Over
  Budget**. The comma's own time and the cable come out of the same 50 ms.
- **Latency** is the average frame, split into Input, Model, Other and Send,
  each measured against the budget.
- **History** has a bar for every 5 seconds of the last 2 minutes, as tall as
  that span's slowest frames.
- **Link** has the frame rate (the comma sends 20 a second) and slow frames,
  and says whether the comma is on the USB cable or an Ethernet adapter.
- **iPhone** has the phone's temperature, battery, memory and link. A hot phone
  slows down, and it shows here before it shows in the ring. Memory is what
  iOS still lets the app use; it turns orange under 1 GB, where preparing a
  model may not fit.

If the app leaves the screen while serving, an orange banner at the top says
so when you come back: iOS suspends an app that is not in front.

On its side the phone shows the ring and the latency with nothing else on
screen. On the other tabs, the state stays in view in a bar above the tabs;
tap it to go back.

## Logs

**Settings > Help > Logs**, or the document button on Status, shows what the
server and the app have logged: connections, model preparation, slow frames,
memory pressure, heat, and the app going to the background. Warnings are
orange and errors red. The share button sends the whole text; **Clear**
empties the view.

## Heat

A phone running a model 20 times a second warms up, and a phone in a mount in
the sun warms faster. When iOS decides it is too hot it slows the chip down,
and frames start missing 50 ms. In one drive of about 30 minutes on an earlier
build the phone slowed as it warmed. The 10-minute benchmark shows how your
phone and mount fare; the Temperature tile and the logs say when it throttles
on the road. Keep the phone out of the sun, and out of a thick case.

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
| Link | Whether the comma is on the USB cable or an Ethernet adapter, or that the phone is dialing |
| Port | The TCP port the phone listens on, 5599 by default. Over the cable the phone dials the comma's port instead; the comma's `JetlinkEndpoint` names this one |
| USB, Ethernet, Wi-Fi | The phone's addresses. The USB one is the comma's lease over the cable; the Ethernet one is what `JetlinkEndpoint` should say |
| Compute | Neural Engine (default): the whole model on it. Neural Engine + GPU: the Mac's layout, the vision trunk on the Neural Engine and the rest on the GPU. GPU: for when another app keeps the Neural Engine busy. Changing it prepares the model again |
| CPU Keep-Warm | On by default. A CPU core kept busy between frames while the Neural Engine runs the model, so the next frame is not waiting on a core that went to sleep. It uses some power |
| GPU Keep-Alive | A small GPU job between frames so the GPU does not slow down in the gaps. It uses some power |
| Keep Screen On | On by default. With it off, auto-lock suspends Jetlink |
| Help | How to connect the comma, and the logs |
