# Cables, networking, and power

Initial setup: [README](../README.md#quick-start), [Jetson
guide](jetson.md), [platform setup](platforms.md).

## USB connection

A USB 3 data cable; charge-only cables do not work.

| Server | Cable to the comma's USB-C port |
| --- | --- |
| Jetson | USB-A to USB-C, from the Jetson's USB-A port (its USB-C port may not connect) |
| Mac | USB-C cable, or USB-A to USB-C with a USB-C adapter |
| Linux PC | USB-A to USB-C, from a USB-A port on the PC |
| iPhone | USB-C cable, or USB-A to USB-C with a USB-C adapter; a powered USB-C hub between them keeps the phone charging |
| Android | USB-A to USB-C, from a USB 3 hub with USB-C power pass-through on the phone, so it charges; or a USB-C to USB-A adapter |

- The comma holds its USB-C port as the device for any host but a chestnut.
- The comma's USB-C port cannot serve Jetlink and chestnut at once.

### What the comma presents

One of two USB gadgets, per the comma's **Accelerator Link** setting (models
settings):

| Setting | For | Gadget |
| --- | --- | --- |
| **USB** | Jetson, Mac, Linux PC, Android | Plain: one vendor-specific interface, one bulk endpoint pair, opened through usbfs on Linux (IOKit on the Mac; usbfs on the descriptor Android's USB host API hands the app). No network interface. |
| **iOS** | iPhone | Composite: interface 0 is the same vendor interface (never used on iOS), then a CDC-NCM network interface, since iOS gives apps no vendor USB access but drives USB network adapters itself. |

- iOS network: the comma is `192.168.60.1` and runs DHCP; the phone gets a
  `192.168.60.x` address with no gateway or DNS, keeps its internet route over
  Wi-Fi, and dials `192.168.60.1:5599`.
- Changing the setting rebuilds the gadget (an unplug), so it changes only
  offroad.
- Comma side: the `jetlink.comma` package. The owner holds the gadget and lends
  modeld its endpoints or the phone's dial; every root step goes through
  `scripts/comma/jetlink-root.sh`. See the
  [installation reference](installation-reference.md#custom-usb-integrations).

### Bus speed

Latency needs USB 3 (SuperSpeed). A frame is about 400 KB to the server and
8 KB back (the model's hidden state stays on the server): about 1 ms on USB 3,
10 ms on USB 2 (hence USB 3 on every hop: cable, adapter, any hub).

On Linux the server turns off USB 3 link power management on the comma's port:
waking the link from its low-power states cost 3.9 of the 7.6 ms each frame
spent in transport on the bench Jetson.

Negotiated speed on the comma: `/sys/class/udc/*/current_speed`
(`super-speed` is USB 3, `high-speed` USB 2), also printed with the built
gadget by `sudo scripts/comma/jetlink-root.sh check`.

### The network link on a Linux host

The comma's kernel (4.9, Qualcomm's u_ether) sends NCM blocks slowly when the
host lets it pack several packets into one. Comma to host, bench mici,
SuperSpeed, Jetson host:

| NCM block size | Throughput |
| --- | ---: |
| 16 KB (Linux default) | 22 Mbit/s |
| 2 KB | 190 Mbit/s |

Host to comma is unaffected (340 Mbit/s); CPU is not the limit. A Linux host
using the network link (a bench standing in for a phone, or a PC over the cable
network) should cap the block size:

    echo 2048 > /sys/class/net/<interface>/cdc_ncm/rx_max

`scripts/99-jetlink-host.rules` does that on plug-in. Apple's NCM driver picks
its own block size and showed no slow path (reference phone: 393 KB up in under
19 ms).

With the cap, parked live bench on the comma (Cinque Terre V3, 180 s, 3,416 big
frames, every frame delivered):

| Link | p50 | p99 |
| --- | ---: | ---: |
| Network link | 36.4 ms | 40.6 ms |
| Vendor interface | 28.6 ms | 30.3 ms |

The ~8 ms gap is all in the comma's send of the 393 KB frame (`bench_link.py`:
26 ms of transport vs 8.6 ms). That is the network link's floor on this kernel;
the vendor interface stays the link for every host that can open it.

## Power requirements

Power the Jetson and comma separately. The Jetson's supply and cable must
deliver at least 25 W and tolerate voltage drops at engine start. **Always on**
or **Switched**: see the [power setup table](jetson.md#1-choose-your-power-setup).

### Recommended Jetson power setup

Orin Nano Super devkit: a **straight 12 V-to-DC barrel adapter** on a supply
that **stays on when the ignition is off** (such as an always-on 12 V accessory
socket), with Jetson **deep sleep** enabled.

<img src="images/jetson-12v-dc-adapter.jpg" width="320" alt="Example of a 12 V car accessory socket plug to DC barrel adapter cable">

- Plug: **5.5 mm outer / 2.5 mm inner, center-positive**. Check this when
  buying; the photo shows the style only. Other carrier boards may differ.
- Confirm the socket stays powered after ignition off, including after any
  delayed shutoff.
- Choose **Always on** in the installer for deep sleep; `jetlink setup` changes
  an existing setup.

<a id="always-on-supply-and-suspend"></a>

<a id="powering-off-with-the-comma"></a>

Parking, starting, and how battery-protection shutdown differs from sleep:
[Choose your power setup](jetson.md#1-choose-your-power-setup).

## TCP

The comma links over USB only (the plain gadget, or its network interface for
an iPhone). `jetlink-server --listen` tests a server without a comma: no client
authentication (trusted network only), and Wi-Fi misses the 50 ms frame budget.
See [test without a comma](platforms.md#test-without-a-comma).

<a id="custom-usb-integrations"></a>

Custom USB setups: [installation reference](installation-reference.md#custom-usb-integrations).
