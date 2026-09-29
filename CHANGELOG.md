Jetlink v0.7.1
==============
**Driving**
* **Stays Engaged:** If the big model drops or falls behind while engaged, the comma shows **TAKE CONTROL** for 5 seconds and keeps driving on the small model (was a soft disable). Lateral-only (MADS) driving gets the same warning.
* **No Lag Alert After a Pull:** The small model takes over within a frame, the first pull of a drive included (was over a second), with no "Driving Model Lagging" afterwards. A server that stops answering is caught in 0.2 s (was 0.5).
* **Switch Without Stopping:** Connected mid-drive? Turn cruise fully off and engage again: the next engagement uses the big model. The comma says **Big Model Ready: Re-engage to switch** once, not at every stop.
* **Replug:** A Jetson plugged back in reconnects at once (could wait up to a minute).

**General Updates & Fixes**
* **Accelerator Link:** Switching out of iOS no longer needs a power cycle: change it with the car off.
* **Update the comma:** zoompilot's `jetson-trt` branch pins Jetlink v0.7.1. All of this runs on the comma; the server needs no update.

Jetlink v0.7.0
==============
**Android support!**
* Run big models on a Snapdragon phone's NPU over USB (experimental, build from source) <3
* Turn on: Settings > Models > Accelerator Link > USB.

**One Swift engine everywhere**
* Jetsons, Linux PCs, the Mac, iPhone/iPad and Android all run the same Swift server.
* On a Jetson it gives v0.6.0's output bit for bit, with half the CPU.
* **No Docker:** Jetsons and PCs run it natively. `jetlink update` moves your install over, keeping settings, models and engines.
* **Update the comma and Jetlink together:** New link protocol; zoompilot's `jetson-trt` branch pins Jetlink v0.7.0. A comma on an older build stays on the small model.

**General Updates & Fixes**
* **Faster Link:** Big-model frames 3 to 5 ms quicker on the bench Jetson: 2.2 ms of USB transport (was 8.5). Replies are 8 KB (was 74 KB): the hidden state stays on the server. USB 3 power saving is off only while the comma is sending.
* **Parked:** The Jetson sleeps between its half-hourly wake checks: awake about 1% of a park (was 6%).
* **Reliability:** The comma restarts Jetlink if it stops, alerts when it cannot, and no longer stalls at shutdown. A new model the Jetson has never seen uploads right away. "Power off with the comma" reaches the Jetson every time.
* **Forks:** openpilot forks plug Jetlink in through one adapter module (developers: `jetlink/openpilot`).
* **Status Page:** Watch the server from your phone at `http://<name>.local:5600`.
* **Stay Awake:** `jetlink caffeinate` keeps a Jetson awake while you work on it.
* **iPhone:** Neural Engine + GPU is the default (14 ms a frame on an iPhone 18 Pro).
* **Linux PCs:** The installer puts NVIDIA's TensorRT 11.3 in `/opt/jetlink`; no system TensorRT.
* **Tested:** JetPack 7.2.1 and the Mac, in the car. JetPack 6.2, Linux PCs and WSL2 are untested.

**Removed**
* Python server, Docker images and `scripts/run-mac.sh` (use `jetlink-server`).

Jetlink v0.6.0
==============
**Mac**
* **Swift only:** No bundled Python; 14 MB download (was 120 MB).
* **Settings:** Server and Log level removed.
* The Python server still runs from a checkout: `scripts/run-mac.sh`.

Jetlink v0.5.0
==============
**iPhone & iPad support!**
* Run big models on an iPhone or iPad over one USB cable (experimental, build with Xcode) <3
* Turn on: Settings > Models > Accelerator Link > iOS.
* Thank you Casey (@ScriptDrifter) for the original iPhone port!

**One Swift engine**
* The Mac and iPhone/iPad apps now share one Swift server, matched to the Python server's output.

**General Updates & Fixes**
* **Accelerator Link:** Now Off, USB or iOS. Set it again after updating.
* **Default Model:** Cinque Terre V3.
* **Model Switching:** New models download and build in one step.
* **Reliability:** Recovers in seconds if the Jetson server restarts mid-drive.
* **Jetson Power Off:** Fixed the comma not shutting the Jetson down.
* **Installer:** Installs the latest release; `jetlink update` moves to the newest.
* **Mac:** Swift server option with no Python, and a Benchmark page.

**Removed**
* tinygrad backend and Jetson over Ethernet.

Jetlink v0.4.3
==============
**Mac**
* **Fixed:** "No bundled Python runtime" when opening from Finder or the Dock on macOS 15.

Jetlink v0.4.2
==============
**Mac**
* **Signed & Notarized:** Opens without the Gatekeeper workaround.

Jetlink v0.4.0
==============
**One-command install!**
* Sets up your Jetson or Linux PC with one command.
* `jetlink update` updates, and rolls back if it fails.

**General Updates & Fixes**
* **JetPack:** 6.2 and 7.2 (7.2 is ~12% faster).
* **Linux PCs:** NVIDIA driver 580+.
* **Models:** Cinque Terre V3, and new big models show up without a Jetlink update.
* **Jetson Sleep:** Fixed on stock JetPack.

**Mac**
* **Apple Silicon:** M1 Pro and up, with the model split across the Neural Engine and GPU.
* **Models:** Preload with one click. Models from v0.3.0a1 prepare again once.
* **Latency:** New visualizations.
* **Branding:** New Jetlink look and icon.
