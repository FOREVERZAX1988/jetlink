# Jetlink for iPhone

**Experimental, and not yet run on a phone.** Jetlink for iPhone runs the
Jetlink server inside an iPhone app and serves the comma over wired Ethernet.
Everything in it has been tested on a Mac, where the same Swift server runs
Cinque Terre V3 at 30 ms per frame. How fast an iPhone runs the model, and
whether it keeps up in a warm car, is still unmeasured.

For a Mac, see [Jetlink for Mac](macos-app.md). For a Jetson or PC, see the
[README](../README.md#quick-start).

## How it differs from the Mac app

| | Mac | iPhone |
| --- | --- | --- |
| Server | The Python server, in an embedded runtime | A Swift port of the server core, in the app |
| Runtime | onnxruntime 1.29 with CoreML | The same: onnxruntime 1.29 with CoreML, the same options |
| Model layout | Vision on the Neural Engine, the rest on the GPU | The same, prepared byte for byte as the Python server prepares it |
| Link to the comma | USB (the comma is a USB gadget) | TCP over Ethernet. iOS gives apps no raw USB access |
| While driving | Runs in the background | Must stay on screen. iOS suspends a background app |

## Requirements

| What you need | Why |
| --- | --- |
| An iPhone with USB-C on iOS 26 | For a USB-C Ethernet adapter. A Pro model's USB 3 port leaves more room in the frame budget than USB 2 |
| Two USB-C gigabit Ethernet adapters and a short Ethernet cable | One adapter on the comma, one on the phone. The comma supports Realtek RTL8152/8153 and ASIX AX88179 adapters |
| An adapter with USB-C power passthrough on the phone side | The phone runs the model 20 times a second and belongs on power |
| About 3 GB of free space per model | A 766 MB download plus the prepared CoreML engine |
| A Mac with Xcode 26 and the iOS 26 platform | There is no App Store or TestFlight build; you build and install it yourself |

You also need a comma running a zoompilot build with Jetlink, set up as in the
[README](../README.md#comma-setup-all-platforms).

## Install

1. In Xcode, open **Settings > Components** and install the **iOS 26**
   platform if it is missing. Xcode cannot build for iPhone without it.
2. From a checkout, run `make -C ios open` to open the project in Xcode.
3. Select the **Jetlink** target. Under **Signing & Capabilities**, choose your
   team and, if you need to, change the bundle identifier. The app asks for the
   Increased Memory Limit capability, as headroom for preparing a model; if your
   account cannot sign it, remove it from `ios/project.yml` and run
   `make -C ios project`.
4. Connect the iPhone, select it as the run destination, and click **Run**.

For building and testing without a phone, see [iPhone development](../ios/README.md).

## Connect the comma

1. Plug an Ethernet adapter into the comma's USB-C port and another into the
   iPhone, and join them with the cable.
2. On the iPhone, open **Settings > Ethernet**, choose the adapter, set
   **Configure IP** to **Manual**, and give it an address such as `10.0.0.2`
   with subnet mask `255.255.255.0`.
3. Give the comma's adapter an address on the same subnet, such as `10.0.0.1`.
4. Set the comma's `JetlinkEndpoint` parameter to the address the app shows,
   such as `10.0.0.2:5599`. See [Ethernet (TCP)](transport.md#ethernet-tcp).
5. Open Jetlink. The first time, allow **Local Network** access when iOS asks.
   Without it iOS can refuse the comma's connection, and the dashboard says so.

Wi-Fi works for testing, including the iPhone's Personal Hotspot, but it does
not meet the 50 ms frame budget. Settings lists every address the phone has,
with Ethernet first.

## Prepare a model before you drive

Open **Models** and tap **Use** on the model your comma uses. Jetlink downloads
it, prepares it for this iPhone and loads it, in one step. The download needs
Wi-Fi or cellular data. Keep Jetlink open until it finishes, because iOS
suspends background apps and their downloads.

If you skip this, the comma sends its model when it connects. It drives on its
small model until the phone has the model ready.

## The dashboard

Mount the phone where you can see it, in either orientation. The screen stays
on while Jetlink is open.

- **The headline** says whether the comma is getting the big model: **Serving**,
  **Waiting for the comma**, **Preparing**, or what is wrong.
- **The ring** is the 50 ms frame budget. It is filled to the slowest 1% of
  frames (p99) over the last ten seconds, and the number inside is the room
  left. Green means room to spare, orange means under 10 ms left, and red means
  over budget. The phone's own time counts toward the 50 ms, and so do the
  comma's time and the network.
- **Where the Time Goes** splits a frame into staging the inputs, running the
  model, overhead, and sending the reply, on the same 50 ms scale.
- **Last Two Minutes** shows the mean and p99 per second, and marks any second
  with a frame over budget.
- **Rate, slow frames, model time and frames** give the numbers behind the
  ring. The comma sends 20 frames per second.
- **Temperature and battery** are the phone's own. A hot phone slows down,
  and it shows up here before it shows up in the ring.

## Limits

- **Keep Jetlink on screen.** iOS suspends an app that is not in front, or
  whose phone locks. The comma then drives on its small model, and if engaged
  it asks you to take over. Do not use other apps on the phone while driving.
- **Heat.** A phone in a mount in the sun throttles. Watch the temperature tile.
- **The comma cannot power the phone off.** A shutdown request is refused.
- **One comma at a time.** A new connection replaces the current one.

## Settings

| Setting | What it does |
| --- | --- |
| Port | The TCP port the comma's `JetlinkEndpoint` names, 5599 by default |
| Run the Model On | Neural Engine and GPU (default), or GPU only if another app keeps the Neural Engine busy. Changing it prepares the model again |
| Keep the GPU Clocked Up | A small GPU job between frames so the GPU does not slow down in the gaps. It uses some power |
| Keep the Screen On | On by default. With it off, auto-lock suspends Jetlink |
