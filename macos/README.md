<a id="jetlink-for-mac"></a>

# Jetlink for Mac development

To install and use the app, follow the [Mac user guide](../docs/macos-app.md).
This page covers building, testing, and signing it from source.
Run the commands from the repository root.

## Build

```
brew install xcodegen
make -C macos app
```

`make app` generates the Xcode project, builds Release for Apple silicon, and
signs the app ad hoc. The result is `macos/build/Jetlink.app`. The server is
the Swift one in `JetlinkKit` (every platform runs the same code), linked into
the app with onnxruntime's static xcframework, which Xcode downloads (61 MB)
the first time it resolves the package.

Check the built app with `make -C macos smoke`: it launches the bundle with TCP
on a free port and a cache of its own, says hello and pings over the wire
protocol, and stops it. Any `python3` runs the client; it uses only
`jetlink/protocol.py` and the TCP transport from this checkout.

## Develop

```
make -C macos dev
```

This opens `Jetlink.xcodeproj`. The Swift server itself lives in
`JetlinkKit/Sources/JetlinkServer`; `swift test --package-path JetlinkKit`
runs its tests, and `jetlink-server` runs it without the app
([from a terminal](../docs/platforms.md#from-a-terminal)).

`make -C macos project` regenerates the project from `project.yml` alone;
`make -C macos test` runs the Swift Testing suites. The test host starts no
server. The generated `Jetlink.xcodeproj` is committed, so
`open macos/Jetlink.xcodeproj` works without xcodegen installed.

## Sign, notarize, release

For CI releases and signing secrets, see [publishing](../docs/publishing.md).

```
SIGN_IDENTITY="Developer ID Application: Name (TEAMID)" make -C macos app
NOTARY_KEY_ID=... NOTARY_ISSUER_ID=... NOTARY_KEY_PATH=AuthKey_XXXX.p8 make -C macos notarize
SIGN_IDENTITY="Developer ID Application: Name (TEAMID)" make -C macos dmg
```

`SIGN_IDENTITY` defaults to `-` (ad hoc), which is all a machine without a
Developer ID certificate can do; such a build runs locally but Gatekeeper will
not accept it on another Mac. The hardened runtime is on either way.

## Outputs

Everything lands in `macos/build/`: `Jetlink.app`, `DerivedData/`, and from
`make dmg` the `Jetlink-<version>-macOS.dmg` and `SHA256SUMS` (plus a `.zip`
for an ad hoc build). None of it is committed.

The app icon is generated once by `scripts/make-icon.swift` and its output is
committed under `Resources/Assets.xcassets`.
