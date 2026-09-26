# Cables, networking, and power

For initial setup, see the [README](../README.md#quick-start), [Jetson
guide](jetson.md), or [platform setup](platforms.md).

## USB connection

Use a USB 3 A-to-C data cable. Charge-only cables do not work.

| Server | Connection to the comma's USB-C port |
| --- | --- |
| Jetson | USB-A port on the Jetson |
| Mac | USB-A port on a hub, dock, or USB-C-to-A adapter |
| Linux PC | USB-A port on the PC |

Use the USB-A connection shown above. The Jetson's USB-C port and a direct
C-to-C cable on a Mac may not connect correctly. The comma's USB-C port cannot
serve Jetlink and chestnut at the same time.

## Ethernet (TCP)

Use wired Ethernet for TCP. On the comma, use a USB-C gigabit Ethernet adapter
with a Realtek RTL8152/8153 or ASIX AX88179 chipset. These are supported by
the comma.

1. Connect the comma and server to a wired network with fixed IP addresses.
2. Start the server with `--transport tcp`.
3. Set the comma's `JetlinkEndpoint` parameter to `<server-ip>:5599`, replacing
   `<server-ip>` with the server's wired-network IP address.

TCP has no client authentication. Use a trusted network. Wi-Fi exceeds the 50 ms
frame budget; use USB 3 or wired Ethernet.

To test a server without a comma, follow [test without a
comma](platforms.md#test-without-a-comma).

## Power requirements

Use separate power supplies for the comma and server. Size the Jetson supply for
its 25 W power mode. The comma's USB port cannot power the Jetson.

The supply must tolerate voltage drops when the engine starts. A voltage drop
can reboot the Jetson and interrupt the link. With ignition-switched power,
allow about 65 to 96 seconds from power-on until the model is ready. The comma
uses its small model during startup. See [daily use](using-jetlink.md#what-to-expect-when-driving)
for when it switches to the large model.

### Recommended Jetson power setup

For the Orin Nano Super devkit, we recommend a **straight 12 V-to-DC barrel
adapter**, connected to a supply that **stays on when the ignition is off**,
with Jetson **deep sleep** enabled. An always-on 12 V accessory socket and an
adapter like the one below make this straightforward.

<img src="images/jetson-12v-dc-adapter.jpg" width="320" alt="Example of a 12 V car accessory socket plug to DC barrel adapter cable">

For the Orin Nano Super devkit, use a **5.5 mm outer / 2.5 mm inner,
center-positive** plug. Check these specifications when buying; the photo
shows the adapter style. Other Jetson carrier boards may have different
power requirements.

1. Confirm the 12 V source stays powered after the car is off, including after
   any delayed accessory-power timeout. The Jetson needs power throughout sleep.
2. Connect the adapter to the Jetson's DC input. The socket and cable must
   support the Jetson's full running power, as described above.
3. Choose **Always on** in the installer, or run `jetlink setup` to change an
   existing installation. This configures deep sleep and USB wake.

<a id="always-on-supply-and-suspend"></a>

### When you park and start the car

With the Jetson connected to **always-on power** and **Always on** selected
in the installer:

| When | What the Jetson does |
| --- | --- |
| Ignition off | Goes into deep sleep after a few minutes. Leave its power and USB cables connected. |
| Car started | The comma wakes the Jetson automatically over USB. You do not need to press its power button. |

Deep sleep uses about **300 mW (0.3 W)** directly on 12 V. Actual consumption
varies with your supply and accessories.

### Powering off with the comma

The installer also asks whether the comma may shut down the Jetson to protect
the car battery. If enabled, the Jetson turns fully off when the comma shuts
down for low battery or after a long time parked.

**After a full shutdown, starting the car will not restart a Jetson connected
to always-on power.** Press the Jetson's power button or disconnect and
reconnect its power. The comma can wake it from deep sleep, but cannot turn it
back on after a full shutdown.

You can change this setting with `jetlink setup`. See the
[technical reference](jetson-power-reference.md) for sleep timing, shutdown
thresholds, and custom power setups.

<a id="custom-usb-integrations"></a>

For custom USB setups, see the [installation reference](installation-reference.md#custom-usb-integrations).
