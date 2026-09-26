# Jetson setup

Jetlink runs on a Jetson Orin in the car, connected to the comma by a USB cable.
Three steps: put JetPack on the Jetson, run the installer, and set up the comma.
Allow about an hour the first time, mostly downloads.

## What you need

- A **Jetson Orin**, such as the Orin Nano Super Developer Kit (8 GB).
- **JetPack 7.2.1** (recommended) or **6.2**.
- A microSD card of 64 GB or more, or better an NVMe SSD.
- A power supply that can deliver the Jetson's full power: 25 W or more for an
  Orin Nano. We recommend a straight **12 V-to-DC barrel adapter** from a source
  that **stays on with the ignition off**, paired with deep sleep. For the Orin
  Nano Super devkit, use a **5.5 mm outer / 2.5 mm inner, center-positive** plug.
  See the [adapter example and power setup](transport.md#recommended-jetson-power-setup).
- Internet during setup, and a **USB 3 A-to-C data cable** for the comma.

## 1. Put JetPack on the Jetson

Skip this if the Jetson already runs JetPack 7.2 or 6.2. To check, run
`cat /etc/nv_tegra_release` on it: `R39` with `REVISION: 2.1` or later is
JetPack 7.2.1, and `R36` with `REVISION: 4.3` or later is JetPack 6.2.

JetPack 7.2 is installed from a USB stick and **erases the drive you install it
on**. NVIDIA's [Orin Nano quick start
guide](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/quick_start.html)
has the details; in short:

1. On another computer, download the **Jetson ISO** from NVIDIA's [JetPack
   page](https://developer.nvidia.com/embedded/jetpack) and write it to a USB
   stick of 16 GB or more with [balenaEtcher](https://etcher.balena.io/).
   Copying the file onto the stick does not work.
2. Connect a DisplayPort monitor, a keyboard, the target microSD card or SSD,
   and the USB stick to the Jetson, then power it on.
3. Press **Esc** at the NVIDIA logo, open **Boot Manager**, and choose the USB
   stick.
4. **Press Y within 30 seconds** when it offers a firmware update. Missing this
   is the most common reason the install fails. The Jetson may restart on its
   own afterwards; that is expected.
5. Choose **Install Jetson ISO**, pick the drive, and confirm.
6. Remove the USB stick when it finishes, then go through the first-boot setup:
   language, network, and your user account.

A Jetson that came with very old firmware (older than 36.0) needs NVIDIA's
JetPack 6 update path first; the quick start guide explains how to check.

## 2. Run the installer

On the Jetson, open a terminal (or connect with `ssh`) and run:

```bash
curl -fsSL https://raw.githubusercontent.com/zoompilot/jetlink/main/install.sh | bash
```

It shows what it found, asks how the Jetson is powered, shows the plan, and asks before
changing anything. Choose the power option that matches your wiring.

| It asks | What it means |
| --- | --- |
| How is the Jetson powered in the car? | Choose **Always on** if power stays on with the ignition off. The Jetson sleeps when parked and the comma wakes it when you start the car. Choose **Switched** if power turns off with the ignition; the Jetson boots each time you start the car. |
| Allow the comma to shut down the Jetson to protect the car battery? | If enabled, the comma can turn the Jetson fully off for battery protection. After that, you must press the Jetson's power button or disconnect and reconnect its power. |

See [parking and starting the car](transport.md#always-on-supply-and-suspend)
for what to expect with always-on power and deep sleep.

When it finishes, it prints the comma steps. Running it again is safe: it
offers to keep your answers and brings everything up to date.

## 3. Set up the comma

Follow [comma setup](../README.md#comma-setup-all-platforms) in the README:
after resetting the comma, enter **`zoompilot/jetson-trt`** as the install URL.
If you already use zoompilot, you can select **jetson-trt** in its branch settings
instead. Turn on **Accelerator Link**, then connect the comma's USB-C port to
one of the Jetson's **USB-A** ports. After download and
transfer, the default model takes about 3 minutes to prepare. Wait for the comma's home-button icon to turn green before leaving
the setup. If it stays orange or never pulses, use the checks below.

## Everyday use

```bash
jetlink status      # is it running, is the comma connected, which power setup
jetlink logs        # follow the server's log; Ctrl-C to stop watching
jetlink restart     # restart the server
jetlink update      # get the newest Jetlink, keeping your answers
jetlink setup       # answer the questions again, for example after rewiring the Jetson's power
jetlink uninstall   # remove Jetlink; asks before deleting downloaded models
```

The commands ask for your password when they need administrator rights.

On switched power, allow 65 to 96 seconds from power-on until the model is
ready. The comma uses its small model until it can switch. See
[daily use](using-jetlink.md) for the switching conditions and icon meanings.

## Choosing a model

Start with the default. To change it, open **Settings > Models > Big Model**
on the comma while parked and online. Prefer the 766 MB models on Jetson;
[larger models leave little timing margin](status.md#measured-performance).

### Downloading a model on the Jetson

You can use the Jetson's internet connection to download and prepare a model
before connecting the comma. Follow [model management](models.md#on-a-jetson-or-an-installed-pc).
This is optional; the comma normally sends the model automatically.

## Troubleshooting

| Problem | What to do |
| --- | --- |
| The installer stops with an error | Run it again: it is safe to repeat, and picks up where it left off. The full log is in `/var/log/jetlink-install.log`. |
| The icon never pulses, the server keeps waiting | `jetlink status` should say running. Use a Jetson USB-A port, and try another USB 3 data cable. |
| Engine build fails | Check free disk space (`df -h /mnt/data`) and `jetlink logs` |
| Large model build is killed or hangs | Check `free -h` shows 8 GB of swap; the installer skips it when the disk is too small |
| Model repeatedly drops out | Check separate supplies and voltage dips, cable, cooling, and `jetlink logs` |
| Frame time exceeds 50 ms | Check USB 3 speed, the power mode (`sudo nvpmodel -q`), cooling, and model choice |
| Jetson fails to wake | See [sleep and wake guidance](transport.md#always-on-supply-and-suspend) |
| Jetson stays off after the comma shut it down | This was a full shutdown, not sleep. Press the Jetson's power button or disconnect and reconnect its power; the comma cannot boot it over USB. |
| Server refuses the comma after an update | Update both sides together, see [updates](releasing.md) |
| "Speed Error: nan" or no path | Stop the test and collect logs |

### Reporting a problem

Include your platform, JetPack version, Jetlink commit (`jetlink status` shows
the server image), selected model, time of the test, the exact alert, and what
you saw or heard. For a drive investigation, share the dongle ID from
**Settings > Device**.

Save this boot's Jetson log (use `-b -1` for the previous boot):

```bash
sudo journalctl -u jetlink-server -b --no-pager > jetson.log
```

For comma logs, enable SSH in **Settings > Device** with your GitHub username to
authorize your GitHub SSH keys. Find the comma's IP in **Settings > Network**.
From a laptop with the matching private key, run the command below. Replace
`<comma-ip>` with the comma's IP address:

```bash
ssh comma@<comma-ip> 'tar czf - /data/log' > comma-log.tgz
```

<a id="what-the-installer-changes"></a>
<a id="installing-by-hand"></a>

For manual installation and system changes, see the
[installation reference](installation-reference.md#jetson-installation).
