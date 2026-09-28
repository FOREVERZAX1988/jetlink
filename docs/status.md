# Performance and operating limits

Jetlink is experimental. If the link drops while engaged, the comma
soft-disables: take over. See [daily use](using-jetlink.md).

<a id="status-and-known-limitations"></a>
<a id="platform-testing"></a>

Requirements: [Jetson](jetson.md), [Mac](macos-app.md),
[iPhone](iphone-app.md), [PC](platforms.md).

## Measured performance

Orin Nano Super 8 GB, TensorRT 10.3 FP16, USB 3, recorded-segment replay:

| Model | GPU inference | Full modeld mean / max | First engine build |
| --- | ---: | ---: | ---: |
| BMRLNAP, 766 MB | 19.8 ms | 31.0 / 32.7 ms | 166 s |
| TGC v2, 766 MB | ~20 ms | 31.1 / 33.5 ms | 166 s |
| Lebowski, 1757 MB | 36.2 ms | 46.3 / 49.5 ms | 290 s |

* The frame budget is 50 ms; Lebowski leaves little margin.
* Sustained use at high temperatures is untested.

Mac numbers: [Mac performance](mac-performance.md).

<a id="what-still-needs-validation"></a>

## Power and connection

* Use separate power supplies for the comma and server. Voltage drops can
  reboot the server and drop the link.
* Keep laptops powered, awake, and cooled.
* Power and sleep details: [power setup](transport.md#power-requirements).
* Link over USB. TCP is for testing only ([TCP](transport.md#tcp)).
