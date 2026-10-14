# QNN HTP stubs: matching a build to a phone's Hexagon

> Scope: the Android **NPU** profile, which runs QNN through onnxruntime. The
> `npu:` lines of a report print both sides of the match
> ([troubleshooting](troubleshooting.md)). Related: [Android app](../android/README.md).
>
> Recorded 2026-10-14, from a Snapdragon whose NPU profile could not start.

QNN reaches a Snapdragon's Hexagon through a pair of libraries named after the
phone's **HTP architecture**, not after the phone:

| library | side | where it lives |
| --- | --- | --- |
| `libQnnHtpV<NN>Stub.so` | app | `jni/arm64-v8a/` inside the APK; a Maven `qnn-runtime` AAR is a directory of them |
| `libQnnHtpV<NN>Skel.so` | DSP | beside the Stub, about 20 MB each; QNN hands it to the Hexagon |

Both halves come from one QAIRT build, because the Stub only speaks to the
architecture it was built for. When the pair for the phone's architecture is
missing, QNN refuses the model in silence: no device is created
(`QNN_DEVICE_ERROR_INVALID_CONFIG: Invalid config values`), the provider takes
no node of the graph, onnxruntime writes no EP context, and the whole model
runs on the phone's CPU -- minutes a frame, under a warm-up that never ends.
The app fails there now, and its `npu:` report names the architectures the
build carries and the phone's own libraries.

## Which `<NN>` is this phone

Qualcomm AI Hub's chipset table maps one to the other: `qualcomm/ai-hub-models`,
`src/qai_hub_models/devices_and_chipsets.yaml`, an entry per chipset carrying
`htp_version` and `soc_model`, with the `sm####` designator among its aliases.
The rows of the phones seen here:

| HTP | Snapdragon | `soc_model` |
| --- | --- | --- |
| 68 | 888 | 30 |
| 69 | 8 Gen 1 | 36 |
| 73 | 8 Gen 2 | 43 |
| 79 | 8 Elite for Galaxy | 69 |
| 81 | 8 Elite Gen 5 for Galaxy | 87 |
| 85 | 8 Elite Gen 6 | 113 |
| 85 | 8 Elite Extreme Gen 6 (an **SM8975**, tried 2026-10-14) | 103 |

Read the architecture off that table rather than off the model number: the
Stub's name depends on nothing else, and the number is what a search turns up.

## What a Maven-only build carries

`com.qualcomm.qti:qnn-runtime` is the artifact `onnxruntime-android-qnn`
depends on, so an APK assembled from Maven gets exactly what that AAR holds.
Maven's newest is 2.51.0 (2026-10-06); its `jni/arm64-v8a/`, read on
2026-10-14:

| architecture | `...Stub.so` | `...Skel.so` |
| --- | --- | --- |
| V68 | 795,440 | 13,136,716 |
| V69 | 795,440 | 13,180,108 |
| V73 | 802,904 | 19,685,168 |
| V75 | 802,904 | 19,664,660 |
| V79 | 802,904 | 19,486,584 |
| V81 | 827,328 | 20,677,132 |

V81 is the last, next to `libQnnHtp.so`, `libQnnHtpPrepare.so`,
`libQnnSystem.so`, `libQnnGpu.so` and the DSP pair (V66). Nothing past it, so
no model, and no setting, makes an Maven-only build serve a V85 phone.

## Where a V85 pair comes from

Not from Maven, and not from this repository: Qualcomm's AI Stack License
allows these libraries inside an app only, so no public repository ships the
V85 pair -- a search finds the build files that name it, not the file.

The full QAIRT SDK ("Qualcomm AI Engine Direct SDK", from Qualcomm's developer
site or Package Manager) lays the pair out as
`lib/aarch64-android/libQnnHtpV85Stub.so` and
`lib/hexagon-v85/unsigned/libQnnHtpV85Skel.so`. Public build files that link the
SDK's own copies name both, in the same version Maven publishes:

| project | what it names |
| --- | --- |
| `mybigday/react-native-qnn-llm`, `android/CMakeLists.txt` | `${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpV85Stub.so`, `${QNN_SDK_ROOT}/lib/hexagon-v85/unsigned/libQnnHtpV85Skel.so`; its README pins QAIRT `2.51.0.260929` |
| `johnmaster/Cross-Backend_LLM_Inference_on_SnapDragon`, op-package Makefiles (same in `MollySophia/rwkv-qualcomm`) | `SUPPORTED_TARGETS = ... hexagon-v81 hexagon-v85 aarch64-android`, `HEXAGON_SDK_ROOT_V85 := hexagon-sdk-6.5.0`, `HEXAGON_TOOLS_VERSION_V85 := 21.0.03` |
| `Lucchetto/onnxruntime-android-qnn-builds`, `config.env` | `QNN_SDK_VERSION=2.51.0.260929` for an onnxruntime built against the SDK rather than the AAR |

The Skel half may already be on the phone: a Qualcomm AICore payload carries a
set of Skels, and on a phone whose SoC is V81 it carried
`libQnnHtpV81Skel.so libQnnHtpV85Skel.so` (`mweinbach/lineage-os-xiaomi17u`,
`docs/v23-install-validation-20260911.md`). The Stub is the app's to bring, so
read the `npu:` report -- what the phone's own `/vendor` and payloads hold --
before fetching anything.

## Keywords to search with

GitHub's code search reads text and not binaries, so search for the names: it
finds the build files that reference the pair, and through them the QAIRT build
that has it.

| what you want | query |
| --- | --- |
| who links the pair out of an SDK | `libQnnHtpV85Stub`, `libQnnHtpV85Skel`, `hexagon-v85` |
| who names a chip's architecture | `htp_version`, `SM8975`, `soc_model` |
| who pins a QAIRT build | `2.51.0.260929`, `QNN_SDK_ROOT`, `QAIRT` |
| which artifact to read yourself | `com.qualcomm.qti:qnn-runtime` and `onnxruntime-android-qnn` on Maven Central |

Name the chip both ways: `SM8975` and *Snapdragon 8 Elite Gen 6* /
*8 Elite Extreme Gen 6*. A query for the architecture alone
(`libQnnHtpV81Stub`) returns the projects of an older phone and shows the shape
of the answer; a query for the newest pair (`libQnnHtpV85Stub`) returns
whichever project got there first.

## Reading a release yourself

```bash
curl -s https://repo1.maven.org/maven2/com/qualcomm/qti/qnn-runtime/maven-metadata.xml | grep '<latest>'
curl -sSO https://repo1.maven.org/maven2/com/qualcomm/qti/qnn-runtime/2.51.0/qnn-runtime-2.51.0.aar
unzip -l qnn-runtime-2.51.0.aar | grep -E 'HtpV.*(Stub|Skel)'
```

An unpacked SDK answers the same question:

```bash
ls "$QNN_SDK_ROOT"/lib/aarch64-android/libQnnHtpV*Stub.so
ls "$QNN_SDK_ROOT"/lib/hexagon-v8*/unsigned/libQnnHtpV*Skel.so
```

## Until a build carries it

The GPU profile serves the model, and the failing build names it. When a
`qnn-runtime` past 2.51.0, or an SDK build, brings the V85 pair, taking it is
the AAR's own layout: the Stub and its Skel in `jni/arm64-v8a/` of the APK,
about 21 MB for the pair.
