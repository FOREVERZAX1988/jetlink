# Jetlink for iPhone development

To install and use the app, follow the [iPhone user guide](../docs/iphone-app.md).
This page covers building and testing it from source. Run the commands from the
repository root.

## Layout

The app is a thin shell over the `JetlinkKit` Swift package, which every
platform shares:

| Module | What it is |
| --- | --- |
| `JetlinkKit` | The control protocol's types, the model list and its store |
| `JetlinkUI` | SwiftUI both apps draw with: the frame budget, the headroom ring, badges |
| `JetlinkONNX` | Reading and preparing a model's ONNX for CoreML, without the onnx package |
| `JetlinkRegistry` | sunnypilot's model catalog, LFS downloads, imports, the cache layout |
| `JetlinkServer` | The server: wire protocol, TCP, session, queues, control |
| `JetlinkORT` | onnxruntime and its profiles (`OrtBackend`, `OrtProfile`): CoreML here |

`ios/Jetlink` holds only what is the phone's own: the dashboard, the Models and
Settings screens, the server's lifecycle in the app, and the phone's health.

## Build

```
brew install xcodegen
cp ios/Config/Local.xcconfig.example ios/Config/Local.xcconfig   # your team and bundle id
make -C ios open
```

Signing comes from `ios/Config/Signing.xcconfig`, and your own team and bundle
identifier from `Local.xcconfig` beside it, which git ignores. Leave Signing &
Capabilities alone: it would write them into the committed project. The
**Jetlink** scheme runs Release, since frame times from an unoptimized build
mean little; **Jetlink Debug** runs Debug for the debugger.

Xcode builds for iPhone only once the iOS platform is installed (Xcode >
Settings > Components). `make -C ios build` compiles for a device without
signing, and needs the same.

Without the platform, `make -C ios typecheck` compiles and links the app's
sources for arm64 iOS through SwiftPM, against the SDK Xcode ships with.

## Test

The server is tested on the Mac, where it is the same code:

```
cd JetlinkKit && swift test
```

The suites check the wire bytes against `jetlink/protocol.py` and run the whole
server on onnxruntime's CPU provider against golden frames
([conformance](../docs/conformance.md)).

To run the Swift server with CoreML against `scripts/bench_link.py` or
`verify_parity.py`, use `jetlink-server`, which listens on TCP port 5599:

```
swift build -c release --package-path JetlinkKit --product jetlink-server
JetlinkKit/.build/release/jetlink-server --cache /tmp/jetlink-cache
python3 scripts/bench_link.py --host 127.0.0.1 --onnx big_driving_supercombo.onnx --rate 20
```

## TestFlight

Uploading a build to App Store Connect: [publishing](../docs/publishing.md#iphone-app-on-testflight).
