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

### What the comma presents

The comma's USB gadget is composite. Interface 0 is the Jetlink link, a
vendor-specific interface with one bulk endpoint pair, which the Jetson, Mac
and Linux PC servers open through libusb. After it comes a CDC-NCM network
interface (CDC-ECM on a kernel without NCM), for an iPhone: iOS gives apps no
access to a vendor USB device, but drives a USB network adapter itself. The
comma is `192.168.60.1` on that network and runs a DHCP server for it, so the
phone gets a `192.168.60.x` address with no gateway and no DNS, keeps its own
route to the internet over Wi-Fi, and dials the comma at `192.168.60.1:5599`.

A Jetson, Mac or Linux PC plugged into the comma also grows a network interface
(named after the gadget, `jetlink`, or `usb0`/`enx...` on Linux) with a
`192.168.60.x` lease. It carries no default route and needs no setup; ignore
it. The link itself is still the vendor interface.

### Bus speed

Latency depends on the link enumerating at USB 3 (SuperSpeed). A frame is about
460 KB: around 1 ms on USB 3 and around 11 ms on USB 2. That is why the cable
must be a USB 3 A-to-C data cable, and why a phone goes through a USB 3 hub.
On the comma, the negotiated speed is in `/sys/class/udc/*/current_speed`:
`super-speed` is USB 3 and `high-speed` is USB 2. `sudo scripts/setup_gadget.sh
--check` prints it along with what the gadget can present.

## Power requirements

Use separate power for the Jetson and comma. The Jetson's supply and cable
must support at least 25 W and tolerate voltage drops when the engine starts.
For help choosing **Always on** or **Switched**, use the
[power setup table](jetson.md#1-choose-your-power-setup).

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

Check that the socket stays powered after the ignition is off, including after
any delayed shutoff. Choose **Always on** in the installer to enable deep sleep.
To change an existing setup, run `jetlink setup`.

<a id="always-on-supply-and-suspend"></a>

<a id="powering-off-with-the-comma"></a>

For what happens when you park or start the car, and how battery-protection
shutdown differs from sleep, see [Choose your power setup](jetson.md#1-choose-your-power-setup).

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

<a id="custom-usb-integrations"></a>

For custom USB setups, see the [installation reference](installation-reference.md#custom-usb-integrations).
