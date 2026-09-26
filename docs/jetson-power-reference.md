# Jetson power management reference

For setup and everyday behavior, see [Jetson power and sleep](transport.md#always-on-supply-and-suspend).
This page covers implementation details and custom installations.

## Deep sleep and USB wake

To enable idle suspend, choose **Always on**, the recommended answer, when the
[installer](jetson.md#2-run-the-installer) asks how the Jetson is powered, or
run `jetlink setup` to change the answer later. The installer arms USB wake on
the Jetson's hubs, grants the container access to `/sys/power`, and starts the
server with `--sleep-after 120`. Suspend requires `deep` support in
`/sys/power/mem_sleep`; the installer checks, and says so when it is missing.

With idle suspend enabled:

1. After ignition turns off, the comma releases the USB connection once the
   engine is ready and at least one minute has passed.
2. The Jetson suspends after 120 seconds without a USB device connection.
3. A USB connection or disconnection wakes the Jetson. If no device connects,
   the server waits another 120 seconds and suspends again.

The comma reconnects when it needs the server. Without `--sleep-after`, the link
stays connected while the comma remains awake after parking.

If suspend fails, the server retries after 10 seconds and doubles the delay
between attempts, up to 5 minutes. Check the server logs if the Jetson stays
awake. The container needs write access to `/sys/power`, and USB wake must be
enabled on the root hubs and onboard hub.

## Battery-protection shutdown

This optional battery-protection action is a **full shutdown, not sleep**.
When enabled, the comma asks the Jetson to power off when the comma shuts down
under its battery policy (11.8 V or 30 hours parked). The installer sets this
up when you allow the comma to shut down the Jetson; it is the host-side
`jetlink-poweroff.path` unit and its service. The server writes a flag in the
models folder; the host service removes the flag and powers off. Flags from
earlier boots are ignored.

For testing, disable this poweroff action by creating the dry-run file:

```bash
touch /mnt/data/jetlink/poweroff-dry-run
```

Remove the file to enable poweroff again:

```bash
rm /mnt/data/jetlink/poweroff-dry-run
```

A powered-off Jetson stays off on an always-on supply, even if the comma starts
again. USB wake only works from sleep. To boot after a full shutdown, press the
Jetson's power button or disconnect and reconnect its power. For automatic
restart, the installation needs a way to do this, such as a low-voltage
disconnect that restores power when the alternator runs, or an ignition-controlled connection to the J14 power-button
input. The devkit starts automatically when DC power returns.

## Adapter specifications

For the Orin Nano Super devkit, NVIDIA lists the 5.5 mm outer / 2.5 mm inner
connector dimensions in its [hardware guide](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/hardware_layout.html)
and center-positive polarity in the [carrier board specification](https://developer.nvidia.com/downloads/assets/embedded/secure/jetson/orin_nano/docs/jetson_orin_nano_devkit_carrier_board_specification_sp.pdf).
