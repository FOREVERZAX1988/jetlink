# Using Jetlink

After [setup](../README.md#quick-start), leave the server running and connected
to the comma.

- Keep laptops powered and awake; sleep drops the link.
- Keep the iPhone app on screen; iOS suspends it otherwise.

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

- The small model drives while the server starts (switched-power Jetson with a
  cached engine: 65 to 96 seconds).
- The large model takes over only when nothing is steering: **at a stop with
  cruise off, or with lateral control off**. Until then the icon is dimmed and
  the comma says **Big Model Available** at every stop. With lateral control
  always on, disengaging alone is not enough.
- **Big Model Ready** chime: it has taken over.
- **Big Model Lost** while engaged is a soft disable: take over. The small model
  drives; Jetlink reconnects and switches back at the next chance.

## Parking and waking a Jetson

With **always-on power** and **Always on** chosen in the installer:

- **Ignition off:** deep sleep after a few minutes, using about **0.3 W** on
  12 V. Leave power and USB connected.
- **Car started:** the comma wakes the Jetson.
- **After a battery-protection shutdown:** press the Jetson's power button or
  reconnect its power. Starting the car won't restart it.

Adapter and installer choices: [power setup](transport.md#recommended-jetson-power-setup).

## Choose a model

- Offroad and online, open **Settings > Models > Big Model**. Start with the
  default.
- The comma downloads your pick; the small model drives while it prepares.
- List empty or out of date? Use **Refresh Model List**.
- Jetson: prefer the 766 MB models; the 1.7 GB Lebowski leaves little margin
  ([measurements](status.md#measured-performance)).
- Optional: [prepare models ahead of time](models.md).

## Update or stop using Jetlink

- Update the comma and server: [updates and rollback](releasing.md).
- Stop: set **Settings > Models > Accelerator Link** to **Off**.

Link never ready? See [troubleshooting](../README.md#if-something-is-wrong).
