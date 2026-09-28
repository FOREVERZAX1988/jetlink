Jetlink v0.5.0
==============
**iPhone & iPad support!**
* Run big models on an iPhone or iPad over one USB cable (experimental, build with Xcode) <3
* Turn on: Settings > Models > Accelerator Link > iOS.

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
