Unreleased
==========
* Mac
  * The Mac app runs the Swift server; no bundled Python, and no tinygrad
    * The download is about 14 MB instead of 125 MB, and there is no Python runtime to find, sign or start
    * A stored tinygrad backend reads as Automatic; the Log level and Python override settings are gone
    * The Python server still runs from a checkout with `scripts/run-mac.sh`
  * The server opens the comma's USB link through macOS's own USB framework, finds the link by its interface class, and waits quietly while the comma has nothing serving the link
  * A Benchmark page, with the iPhone app's verdict
  * `jetlink-serve --usb` runs the Swift server as the USB host from a checkout
* The comma's USB gadget is composite: the Jetlink link plus a USB network interface (CDC-NCM) for an iPhone over one cable
  * A Jetson, Mac or Linux PC plugged into the comma also gets a `jetlink` network interface with a 192.168.60.x address and no gateway; it can be ignored
  * `scripts/setup_gadget.sh --net` sets up the comma's end of that network after a bind, `--check` prints what the comma can present and the negotiated USB speed
  * The server finds the link by its vendor interface class, wherever the gadget puts it
  * `bench_link.py` and `verify_parity.py` take a phone's dial with `--listen`
* iPhone
  * Connects over one cable: the app dials the comma as soon as it has the comma's address, and the title says Connected over USB or over Ethernet
  * Compute defaults to the whole model on the Neural Engine, with a CPU Keep-Warm setting beside GPU Keep-Alive
  * A Benchmark tab: one or ten minutes at the comma's pace, a verdict, the run in ten-second windows with the phone's temperature, and the `bench_link.py` and `verify_parity.py` commands filled in
  * A Logs screen, a Memory tile, log lines for memory pressure, heat and the app going to the background, a banner while the app is not on screen, and an alert when the comma asks to shut down
  * Signing from a git-ignored `Local.xcconfig`, Release when run from Xcode, and a help page on connecting the comma
* Both apps say whether the comma is on USB 3, USB 2 or TCP, and warn on USB 2 (about 10 ms more a frame)
  * The comma's hello names its link and the speed its USB controller negotiated; the server's link event carries it as `medium`

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
