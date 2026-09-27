Unreleased
==========
* The comma's USB gadget is composite: the Jetlink link plus a USB network interface (CDC-NCM) for an iPhone over one cable
  * A Jetson, Mac or Linux PC plugged into the comma also gets a `jetlink` network interface with a 192.168.60.x address and no gateway; it can be ignored
  * `scripts/setup_gadget.sh --net` sets up the comma's end of that network after a bind, `--check` prints what the comma can present and the negotiated USB speed
  * The server finds the link by its vendor interface class, wherever the gadget puts it
  * `bench_link.py` and `verify_parity.py` take a phone's dial with `--listen`

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
