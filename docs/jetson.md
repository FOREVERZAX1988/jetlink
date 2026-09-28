# Set up Jetlink on a Jetson

Allow about an hour for first-time setup, mostly downloads. You can set up
Jetlink outside the car or in the car while offroad. Keep the Jetson and comma
connected to the internet during setup, and keep both devices powered.

## What you need

- A **Jetson Orin Nano Super Developer Kit (8 GB)**.
- A **64 GB or larger microSD card**, or an NVMe SSD.
- A **comma 3X or comma 4** and a **USB 3 A-to-C data cable**.
- Separate power for the Jetson and comma. The comma cannot power the Jetson.

## 1. Choose your power setup

**We recommend always-on 12 V power and deep sleep.** Use a straight
12 V-to-DC adapter with a **5.5 mm outer / 2.5 mm inner, center-positive** plug.
The supply and cable must support at least 25 W.
[See an adapter example](transport.md#recommended-jetson-power-setup).

Choose the installer option that matches your car's power socket:

| Does the socket stay powered with the ignition off? | Choose | What to expect |
| --- | --- | --- |
| Yes | **Always on** (recommended) | Ignition off: the Jetson sleeps after a few minutes. Car started: the comma wakes it automatically. |
| No | **Switched** | Ignition off: the Jetson loses power. Car started: it boots; allow about 1–2 minutes for the large model. |

Check whether your socket turns off a few minutes after parking. If it does,
choose **Switched**. Deep sleep needs power to stay connected and uses about
**0.3 W (300 mW)** directly on 12 V.

**The installer also asks about battery protection.** Choose **Yes** to let
the comma shut down the Jetson for low battery or after a long time parked.
After this full shutdown, press the Jetson's power button or unplug and
reconnect its power; starting the car alone will not restart it on always-on
power. Choose **No** to keep using deep sleep while parked, without the comma
shutting down the Jetson to protect the battery.

<a id="1-put-jetpack-on-the-jetson"></a>
<a id="2-run-the-installer"></a>

## 2. Install Jetlink

The Jetson needs **JetPack 7.2.1** (recommended) or **6.2** first.
If JetPack is already installed, continue to the command below.

<details>
<summary>New Jetson? Install JetPack first</summary>

You need another computer, a USB stick of at least 16 GB, a DisplayPort
monitor, and a keyboard. **Installation erases the selected Jetson drive.**

1. Download the Jetson ISO from [NVIDIA](https://developer.nvidia.com/embedded/jetpack)
   and write it to the USB stick with [balenaEtcher](https://etcher.balena.io/).
2. Connect the monitor, keyboard, storage, and USB stick to the Jetson.
   Power it on, press **Esc**, and select the USB stick in **Boot Manager**.
3. If offered a firmware update, **press Y within 30 seconds**.
4. Choose **Install Jetson ISO** and your target drive. When finished, remove
   the USB stick and complete the on-screen setup.

For older firmware or installation problems, follow
[NVIDIA's setup guide](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/quick_start.html).

</details>

On the Jetson, open **Terminal** and paste:

```bash
curl -fsSL https://raw.githubusercontent.com/zoompilot/jetlink/main/install.sh | bash
```

Answer the power questions using the choices above. The installer handles the
rest. Allow 10–30 minutes and leave it running until it finishes.

## 3. Connect the comma

1. **Install zoompilot.** After resetting the comma, enter
   **`zoompilot/jetson-trt`** as the install URL. If you already use zoompilot,
   select **jetson-trt** under **Settings > Software > Target Branch >
   Non-Prebuilt Branches**. Wait for installation and any reboot to finish.
2. **Enable Jetlink.** Under **Settings > Models**, set **Accelerator Link** to **USB**.
   Leave **Big Model** at its default.
3. **Connect the cable.** Jetson **USB-A** → comma **USB-C**.
4. **Wait for the green icon** on the comma. The model downloads automatically,
   then takes about 3 minutes to prepare the first time.

You're set up. With always-on power, leave both cables connected: the Jetson
sleeps when parked and wakes when you start the car.

The comma uses its small model until the large model is ready. It switches
**at a stop with cruise off, or with lateral control off**. If you hear
**Big Model Lost** while engaged, take over. Read [daily use](using-jetlink.md)
before driving; Jetlink is experimental.

<a id="troubleshooting"></a>

## Need help?

| Problem | Try this |
| --- | --- |
| Installer stopped | Run the install command again. |
| Icon never turns green | Run `jetlink status`. Check the cable uses the Jetson's USB-A port; try another USB 3 data cable. |
| Jetson will not wake | If it fully shut down, press its power button or unplug and reconnect power. Otherwise check **Always on** is selected with `jetlink setup`. |
| Model fails to prepare or keeps disconnecting | Run `jetlink logs`. Check storage space, power, cable, and cooling. |
| Connection fails after an update | [Update both the comma and Jetlink](releasing.md). |

<details>
<summary>Commands and reporting a problem</summary>

<a id="everyday-use"></a>
<a id="reporting-a-problem"></a>

```bash
jetlink status     # check Jetlink and the comma connection
jetlink logs       # view errors; Ctrl-C to stop watching
jetlink restart    # restart Jetlink
jetlink update     # update, keeping your settings
jetlink setup      # change power settings
jetlink uninstall  # remove Jetlink
```

When asking for help, include the output of `jetlink status`, the model name,
the exact alert, and when it happened. The installer log is at
`/var/log/jetlink-install.log`. Save the server log with:

```bash
sudo journalctl -u jetlink-server -b --no-pager > jetson.log
```

</details>

<a id="choosing-a-model"></a>
<a id="downloading-a-model-on-the-jetson"></a>
<a id="what-the-installer-changes"></a>
<a id="installing-by-hand"></a>

[Daily use](using-jetlink.md) · [Model choices](models.md) ·
[Manual installation](installation-reference.md)
