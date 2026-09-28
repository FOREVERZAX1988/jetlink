Jetlink v0.5.0
==============
**iPhone & iPad support!**
* Run big models on an iPhone or iPad over one USB cable (experimental, build with Xcode) <3
* Turn on: Settings > Models > Accelerator Link > iOS.

**One Swift engine**
* The Mac and iPhone apps now share one Swift server, matched to the Python server's output.

**General Updates & Fixes**
* **Accelerator Link:** Now Off, USB or iOS. Set it again after updating.
* **Default Model:** Cinque Terre V3.
* **Model Switching:** New models download and build in one step.
* **Reliability:** Recovers in seconds if the Jetson server restarts mid-drive.
* **Jetson Power Off:** Fixed the comma not shutting the Jetson down.
* **Mac:** Swift server option with no Python, and a Benchmark page.

**Removed**
* tinygrad backend and Jetson over Ethernet.

Jetlink v0.4.3
==============
* Mac
  * Fixed "This build has no bundled Python runtime" when opening the app from Finder or the Dock on macOS 15

Jetlink v0.4.2
==============
* Mac
  * The app is signed with a Developer ID and notarized, so it opens without the Gatekeeper workaround

Jetlink v0.4.0
==============
* Install script that fully sets up your Jetson or Linux PC
* Update to the latest Jetlink with `jetlink update`; a failed update rolls back
* Supports JetPack 6.2 and 7.2 (7.2 is ~12% faster)
* Linux PCs need NVIDIA driver 580+
* Supports Cinque Terre V3 (needs zoompilot v2026.09.25-16+)
* New big models show up without a Jetlink update
* Fixed Jetson not sleeping on stock JetPack
* Mac
  * Supports M1 Pro and above by splitting vision and policy across the ANE and GPU
  * One click to preload models
  * New visualizations showing latency
  * Jetlink branding
  * Models prepared by v0.3.0a1 re-prepare once
