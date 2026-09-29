# Performance and operating limits

Jetlink is experimental. If the link drops while engaged, the comma
soft-disables: take over. See [daily use](using-jetlink.md).

<a id="status-and-known-limitations"></a>
<a id="platform-testing"></a>

Requirements: [Jetson](jetson.md), [Mac](macos-app.md),
[iPhone](iphone-app.md), [Android](android-app.md), [PC](platforms.md).
Tested on a Jetson with JetPack 7.2.1 and on a Mac; JetPack 6.2, Linux PCs and
WSL2 are untested.

## Measured performance

<!--
  TODO(v0.7.0) SWIFT JETSON NUMBERS: fill this table from the Phase 2 A/B of
  the Swift server (live bench on the comma, and the first build per model),
  then delete this comment and the TBD cells.
-->

Orin Nano Super 8 GB, JetPack 7.2.1, TensorRT 10.16 FP16, MAXN SUPER, USB 3:

| Model | GPU inference | Frame on the comma, p50 / p99 / max | First engine build |
| --- | ---: | ---: | ---: |
| Cinque Terre V3 (default), 766 MB | TBD | TBD | TBD |
| BMRLNAP, 766 MB | TBD | TBD | TBD |
| Cinque Terre V2, 766 MB | TBD | TBD | TBD |
| Lebowski, 1757 MB | TBD | TBD | TBD |

* The frame budget is 50 ms; Lebowski leaves the least margin.
* USB transport on the bench Jetson (round trip minus the server's time,
  comma onroad): 7.6 ms p50 with USB 3 link power management on, 3.7 ms with it
  off. The server turns it off.
* Sustained use at high temperatures is untested.

Mac numbers: [Mac performance](mac-performance.md).

<a id="what-still-needs-validation"></a>

## Power and connection

* Use separate power supplies for the comma and server. Voltage drops can
  reboot the server and drop the link.
* Keep laptops powered, awake, and cooled.
* Power and sleep details: [power setup](transport.md#power-requirements).
* Link over USB. TCP is for testing only ([TCP](transport.md#tcp)).
