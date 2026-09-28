Unreleased
==========
* One root script on the comma, `scripts/comma/jetlink-root.sh`, for the comma four and the comma 3X: the gadget (`gadget`, `net`, `check`, `teardown`), the USB-C port hold (`port hold|off`) and the recording VM tuning (`vm apply|restore`), run through `jetlink.comma.root`. It replaces `scripts/setup_gadget.sh` and drops what neither comma needs: the CDC-ECM fallback, the MAC address writes the kernel refuses, and modprobe
* The comma's device layer moved here from the zoompilot fork, as the `jetlink.comma` package: the gadget and the openpilot params it reads, the owner process that holds it, the endpoint lease, the USB-C port hold and the VM tuning. The fork keeps a shim that starts the owner with its provisioning worker. Standard library only, so the owner stays at about 10 MB
* The comma links over USB only: `JetlinkEndpoint`, which sent the link to a server over Ethernet, is gone. The server's TCP transport stays, for testing
* The owner builds the comma's gadget on its first step, so nothing runs once per boot and `scripts/deploy_to_comma.sh` no longer builds one
* The comma's Accelerator Link setting is Off, USB or iOS
  * USB, for a Jetson or a Mac, is the plain gadget as before, lent to modeld at once
  * iOS adds a USB network interface (CDC-NCM) for an iPhone over one cable: `scripts/comma/jetlink-root.sh gadget --ios`, and `net` sets up the comma's end of that network after a bind
  * `check` prints what the comma has built and the negotiated USB speed
  * The cable's DHCP pool is the whole subnet with 10 minute leases
  * The server finds the link by its vendor interface class, wherever the gadget puts it
  * `bench_link.py` and `verify_parity.py` take a phone's dial with `--listen`
* iPhone
  * Connects over one cable: the app dials the comma as soon as it has the comma's address, and the title says Connected over USB or over Ethernet
  * Compute defaults to the whole model on the Neural Engine, with a CPU Keep-Warm setting beside GPU Keep-Alive
  * A Benchmark tab: one or ten minutes at the comma's pace, a verdict, the run in ten-second windows with the phone's temperature, and the `bench_link.py` and `verify_parity.py` commands filled in
  * A Logs screen, a Memory tile, log lines for memory pressure, heat and the app going to the background, a banner while the app is not on screen, and an alert when the comma asks to shut down
  * Signing from a git-ignored `Local.xcconfig`, Release when run from Xcode, and a help page on connecting the comma
* Mac
  * Settings > Server can run the Swift server, the one the iPhone app runs, inside the app with no Python; the Python server stays the default until it is measured with a comma on USB
  * The Swift server opens the comma's USB link through macOS's own USB framework, finds the link by its interface class, and waits quietly while the comma has nothing serving the link
  * A Benchmark page for the Swift server, with the iPhone app's verdict
  * `jetlink-serve --usb` runs the Swift server as the USB host from a checkout
* Both apps say whether the comma is on USB 3, USB 2 or TCP, and warn on USB 2; on the iPhone the title turns orange
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
