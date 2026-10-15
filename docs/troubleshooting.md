# Troubleshooting

**TAKE CONTROL while engaged:** be ready to take over. The comma stays engaged
on the small model. Investigate the connection while offroad.

| Symptom | First step | If it continues |
| --- | --- | --- |
| No **Jetlink** setting | Older builds call it **Accelerator Link**. Check **Settings > Software** for the `main` branch. | Follow your [setup guide](README.md#set-up). |
| Waiting for comma; icon never pulses | Check Jetlink is running and the comma's **Jetlink** setting is **USB** (**iOS** for iPhone or iPad). | Try another USB 3 data cable. Use a USB-A port on Jetson or Linux PC, and a powered hub for a phone. |
| Pulsing icon | Wait while the model downloads and prepares. | Check internet access and [logs](#get-help). |
| Dimmed green icon | Disengage cruise and lateral control fully to let the model switch. | Wait for **Big Model Active**, then engage again. See [daily use](using-jetlink.md#what-to-expect-when-driving). |
| Orange icon or setup alert | Read the alert and check internet access. | Set **Jetlink** to **Off**, then back to **USB** or **iOS**. |
| Model list is empty | Connect to the internet and choose **Refresh Model List**. | In the Mac app, use **Models > Refresh**. |
| **no warp built for this camera** | Update or reinstall the `main` branch. | Include the exact alert when asking for help. |
| Repeated link drops | Check the cable and separate power supplies. | Check cooling and sleep settings; keep the iPhone app on screen. |
| Small model only after an update | Check the release notes for a protocol change. | [Update both the comma and Jetlink](releasing.md) if required. |

Platform-specific help: [Jetson](jetson.md#troubleshooting),
[Mac](macos-app.md#troubleshooting), [Linux PC](linux-pc.md#troubleshooting),
[iPhone and iPad](iphone-app.md#troubleshooting), [Android](android-app.md#troubleshooting).

## Get help

Include your computer or phone model, comma model, driving model, exact alert,
and when the problem happened.

| Platform | Logs |
| --- | --- |
| Jetson or Linux PC | Copy `jetlink status` output and relevant lines from `jetlink logs`. Installer log: `/var/log/jetlink-install.log`. |
| Mac | Open **Logs** in Jetlink. File: `~/Library/Logs/Jetlink/server.log`. |
| iPhone, iPad, or Android | Open **Settings > Help > Logs** and use the share button. |

<details>
<summary>Extra diagnostics for a phone connection</summary>

For repeated fallbacks, also collect the comma's log over SSH:

```bash
tail -n 200 /data/log/jetlink-owner.log
```

For a direct iPhone cable that will not connect, leave it plugged in for
30 seconds, then run over SSH on the comma:

```bash
sudo /data/openpilot/jetlink_repo/jetlink/comma/jetlink-root.sh check
grep 'USB-C' /data/log/jetlink-owner.log | tail -n 40
sudo dmesg | grep -iE 'usbpd|type-?c|swap|weak charger|reverse boost' | tail -n 60
```

Include the cable model and whether the comma restarted. Repeat with the
cable reversed.

When an Android phone's NPU profile never leaves **preparing**, the app's log
answers why by itself: every attempt writes a report under `npu:`, from the
first. A failed profile is tried twice, once by the preparation and once by the
load after it, and both reports are written in full -- paste the one nearest
the failure, or both. (Builds before `cn-<newest>` held the second back with
`read a moment ago, not again for:`, which left a log with no reason in it at
all.) A report writes a build that wrote no compiled context, and a warm-up
over 20 seconds, each write under `npu:` -- and the reason first, now: the
runtime's own lines (onnxruntime's logger, which carries QNN's), the phone's log
around them, the pid this process is, the pair this run is set to use and the
phone's own pair for the same Hexagon, the QNN libraries the app brought and the
ones the phone has, the Skel the DSP will find first, the DSP libraries and
devices, and the chip's properties. Send those lines with the rest. A phone
whose DSP will not take the pair this build carries can be set to run the
phone's own QNN libraries instead, under **Settings > Developer > Phone's Own
QNN** -- the Developer section is hidden until **About > Version** is tapped
seven times, and it appears above **Check for updates** once it is on.
and the report says which pair was in force either way
([QNN HTP stubs](qnn-htp-stubs.md)).

The libraries the app brought come first, with the directory they were read
from: the Android app hands the server `ApplicationInfo.nativeLibraryDir`,
because a probe of its own cannot find the directory -- `/data/app` may be
walked but not read by an app, so a glob over it matches nothing, and Android
carries no `unzip` to read the APK's entries instead. The four libraries QNN
has to agree on with itself (the backend, the system one, and this phone's
Stub and Skel) are hashed there, so a phone holding another QNN build says so
rather than looking like a build that carries nothing. **the neural runtime's
own lines** is what QNN, the DSP and onnxruntime said themselves, whatever
process said it: a provider that takes no node leaves its reason there and
nowhere else.

A phone whose QNN runtime has no HTP architecture for its Hexagon -- a
Snapdragon newer than the last `libQnnHtpV*Stub.so` the app carries, which the
report prints as **the HTP architectures this build carries** -- cannot run an
NPU profile at all: that build fails with the reason, and the GPU profile
serves the model. [QNN HTP stubs](qnn-htp-stubs.md) maps a phone to the
architecture, lists what Maven carries, and says where the missing pair comes
from and what to search for.

</details>
