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

Not from Maven, and not from upstream: Qualcomm's AI Stack License allows
these libraries inside an app only, so no public repository ships the V85
pair -- a search finds the build files that name it, not the file. This fork
carries the pair inside its APK instead (last section).

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

## The V85 pair this fork carries

`android/app/src/main/jniLibs/arm64-v8a/` holds the pair, 14 MB together, out
of QAIRT **2.51.40.260930**:

| file | from the SDK | bytes | SHA-256 |
| --- | --- | --- | --- |
| `libQnnHtpV85Stub.so` | `lib/aarch64-android/` | 875,568 | `8c3636f34ff763d872623172e192b64983e0578d124e5b73f064c28e372f3874` |
| `libQnnHtpV85Skel.so` | `lib/hexagon-v85/unsigned/` | 14,058,420 | `b06247fa38ffad5b8c55a83ff7d4b5500a69282944c0942253fb57b38eaf6b40` |

Both land in `lib/arm64-v8a/` of the APK as the AAR's own pair does, and
`useLegacyPackaging` extracts them to `nativeLibraryDir`, where `libQnnHtp.so`
opens the Stub by name and the CDSP reads the Skel beside it
(`ADSP_LIBRARY_PATH`, set from that directory). The `npu:` report should now
read `the HTP architectures this build carries: ... V79 V81 V85`.

Qualcomm's license allows the pair inside an app only, so it is not upstream's
to carry and not Maven's to publish: this fork's APK carries it, and a build
without those two files still serves every Hexagon up to V81.
`android/app/build.gradle.kts` fails on half a pair, which QNN would swallow.

**One build per Stub, Skel and HTP library.** The pair is 2.51.40, while the
`libQnnHtp.so` and `libQnnHtpPrepare.so` an APK gets from
`onnxruntime-android-qnn` 1.29.0's `qnn-runtime` 2.42.0 are of that build. QNN
wants one build for all three; if the device still refuses the model with the
V85 Stub in place, the next step is that same QAIRT build's `libQnnHtp.so`,
`libQnnHtpPrepare.so` and `libQnnSystem.so` (not the pair alone), and the
`npu:` lines name the version that answered.

## The matched set, and the release that carries it

Taken the next day (2026-10-15): the trio the AAR supplies was replaced by the
same QAIRT build's, so the Stub, its Skel, the backend and the context
preparer are all 2.51.40.260930. The rest of the archive is byte-identical to
the APK it was made from, so a Hexagon up to V81 keeps the pair it had: the
V68 to V81 Stubs and Skels are still the AAR's, beside a 2.51.40 backend.

| file | bytes | SHA-256 |
| --- | --- | --- |
| `libQnnHtp.so` | 4,158,272 | `aaaf6f963ff6d87db516b231182b9be68893fe58498ce510f1cb4ad0c95df0ef` |
| `libQnnHtpPrepare.so` | 88,234,712 | `5d6fa39ff1af44e61fa7b57c047b0ad8c132e7185bcb7a8abe4fec348f4a7c56` |
| `libQnnSystem.so` | 4,125,232 | `199371bf2d3e61188bdfd19fbc6c34273d53ba11794444d85ef60e482807ce54` |

Qualcomm's license keeps these libraries inside an app, so the set is
published as an APK rather than committed: `app-release-v85matched.apk`
(112,698,391 B,
`fb53d34d9dba883376c71e9e024cbd1ee8607467bb6dc28b17f6225333f0f895`, signed
with the release key) sits beside the CI build's `app-release.apk` on the
`cn-204f17c` release, whose own APK is untouched. Its `BUILD_TAG` is
`cn-204f17c`, the tag it is published under, so the app's updater reads it as
current rather than as an update always one behind. The pair's hashes above
are unchanged: this APK carries the pair the commit already recorded.

Two details of those bytes:

- The Skel in a packaged APK is not the file the SDK hands over. AGP processes
  the `jniLibs` ones, and the copy here differs from the repository's
  `libQnnHtpV85Skel.so` in 503 bytes -- its trailing signature region zeroed,
  the shape the AAR's own Skels have. The checkout's file hashes
  `b06247fa38ffad5b8c55a83ff7d4b5500a69282944c0942253fb57b38eaf6b40`; a phone
  that refuses the model with the matched set in place has that verbatim copy
  to try before doubting the build.
- The CI releases are signed with a debug key a runner generates, a different
  one each time (`.github/workflows/release.yml`), where this APK and the
  other hand-built ones share the release key (`043da762...`).

A `README.txt` beside the APK on the release records the same hashes, the base
it was made from and the commands that made it.
