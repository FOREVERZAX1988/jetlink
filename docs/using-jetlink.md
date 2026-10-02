# Using Jetlink

After [setup](../README.md#quick-start), leave the server running and connected
to the comma.

- Keep laptops powered and awake; sleep drops the link.
- Keep the iPhone app on screen; iOS suspends it otherwise.
- The Android app keeps serving in the background while its notification shows.

## Check the comma's icon

| Icon | Meaning |
| --- | --- |
| Pulsing | Downloading, transferring, or preparing the model. Wait. |
| Green, parked | The large model is ready. |
| Green, driving | The large model is active. |
| Green, dimmed, driving | Ready but cannot switch yet (see below). |
| Orange | Preparation failed. Read the home-screen alert. |
| Back to normal after parking | Normal with Jetson deep sleep: the Jetson is getting ready to sleep. |

## What to expect when driving

- The small model drives while the server starts (a switched-power Jetson has
  its prepared model ready about 30 seconds after power-on).
- The large model takes over only when nothing is engaged. Connected while
  you drive engaged, it waits: the icon dims and the comma says **Big Model
  Ready: Re-engage to switch** once. To switch, turn cruise fully off (and
  lateral control, if it stays on without cruise), then engage again. You do
  not need to stop or restart the car, and a stop does not switch while
  lateral control is on. Connected while nothing is engaged, it takes over at
  once.
- **Big Model Active** chime, about a second after the switch: it has taken
  over and you can engage. Inside that second cruise will not engage (**Big
  Model Loading**) while the large model builds its history; lateral control
  turned on then comes on by itself when the second ends.
- **TAKE CONTROL: Big model lost, small model driving** while engaged: the link
  dropped, or the large model fell behind: one frame over 100 ms, two over
  75 ms within 10 seconds, or a second skipped camera frame within about
  6 seconds (a slow phone skips them well before openpilot would disengage
  for **Driving Model Lagging**). openpilot stays engaged on the small model,
  which starts without history: be ready to take over for the next few seconds.
  Jetlink reconnects in the background, at once if you plug the cable back
  in; switch back the same way.

## Parking and waking a Jetson

With **always-on power** and **Always on** chosen in the installer:

- **Ignition off:** deep sleep after a few minutes, using about **0.3 W** on
  12 V. Leave power and USB connected.
- **Car started:** the comma wakes the Jetson.
- **After a battery-protection shutdown:** press the Jetson's power button or
  reconnect its power. Starting the car won't restart it.

Adapter and installer choices: [power setup](transport.md#recommended-jetson-power-setup).
To keep it awake while you work on it, run `jetlink caffeinate`.

## Status page

A read-only page for your phone: the link, model, frame times, hardware and
log of a Jetson or PC.

- **Turn on:** the installer asks for a port (5600; 0 is off). Change it with
  `jetlink setup`.
- **Same network:** in the car, turn on tethering in the comma's network
  settings, then join the Jetson to it once (it reconnects by itself):
  `sudo nmcli dev wifi connect "<hotspot name>" password "<password>"`. Your
  phone's hotspot works the same way.
- **Open:** `http://<jetson ip>:5600`. `jetlink status` prints the address.

## Choose a model

- Offroad and online, open **Settings > Models > Big Model**. Start with the
  default.
- The comma downloads your pick; the small model drives while it prepares.
- List empty or out of date? Use **Refresh Model List**.
- Jetson: prefer the 766 MB models; the 1.7 GB Lebowski leaves little margin
  ([measurements](status.md#measured-performance)).
- Optional: [prepare models ahead of time](models.md).

## Update or stop using Jetlink

- Updating the comma or the server: [updates and rollback](releasing.md).
- Stop: set **Settings > Models > Accelerator Link** to **Off**.

Link never ready? See [troubleshooting](../README.md#if-something-is-wrong).
