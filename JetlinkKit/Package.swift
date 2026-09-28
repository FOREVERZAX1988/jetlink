// swift-tools-version: 6.2
//
// The Swift half of jetlink: the server, and what the apps share. One manifest
// for every platform: what differs is conditioned on the platform being built
// for, and a file for one platform compiles to nothing on the others, so every
// target builds everywhere and every test suite runs wherever it builds.
//
//   JetlinkKit       the control protocol's types and the stores the apps' views read
//   JetlinkUI        SwiftUI pieces the Mac and iPhone apps draw with (Apple only)
//   JetlinkONNX      reading and preparing a driving model's ONNX, without the onnx package
//   JetlinkRegistry  sunnypilot's model catalog, LFS downloads, and the cache layout
//   JetlinkServer    the jetlink server in Swift: the wire protocol, the session, the
//                    queues, the engine host and cache, and the comma's gadget over
//                    IOKit (macOS) or usbfs (Linux, Android). The host passes in the
//                    backend that runs the model, and the gadget.
//   JetlinkORT       onnxruntime's backends: CoreML on Apple, QNN on Android, and the
//                    CPU provider under either
//   JetlinkAndroid   the JNI library the Android app loads, libjetlink.so (Android only)
//   jetlink-serve    the server on its own, for benches
//   jetlink-onnx     the preparation on its own, for checking it against Python
//
// onnxruntime is linked in on Apple platforms and opened at run time elsewhere
// (JL_ORT_DLOPEN): the app's copy from the AAR on Android, the official
// tarball's on Linux. Building for either needs onnxruntime's C headers, with
// -Xcc -I<a directory holding onnxruntime/onnxruntime_c_api.h>;
// android/scripts/swift-build.sh passes the AAR's.
import PackageDescription

let apple: [Platform] = [.macOS, .iOS]
/// No CryptoKit and no onnxruntime framework: swift-crypto, and dlopen.
let linux: [Platform] = [.linux, .android]

let crypto: Target.Dependency = .product(name: "Crypto", package: "swift-crypto", condition: .when(platforms: linux))

let package = Package(
  name: "JetlinkKit",
  platforms: [.macOS(.v15), .iOS(.v26)],
  products: [
    .library(name: "JetlinkKit", targets: ["JetlinkKit"]),
    .library(name: "JetlinkUI", targets: ["JetlinkUI"]),
    .library(name: "JetlinkONNX", targets: ["JetlinkONNX"]),
    .library(name: "JetlinkRegistry", targets: ["JetlinkRegistry"]),
    .library(name: "JetlinkServer", targets: ["JetlinkServer"]),
    .library(name: "JetlinkORT", targets: ["JetlinkORT"]),
    .library(name: "jetlink", type: .dynamic, targets: ["JetlinkAndroid"]),
    .executable(name: "jetlink-serve", targets: ["jetlink-serve"]),
    .executable(name: "jetlink-onnx", targets: ["jetlink-onnx"]),
  ],
  dependencies: [
    .package(url: "https://github.com/apple/swift-crypto.git", "3.0.0"..<"5.0.0")
  ],
  targets: [
    // os.Logger's shape, where there is no os module.
    .target(name: "JetlinkLog"),
    .target(name: "JetlinkKit", dependencies: ["JetlinkLog"]),
    .target(name: "JetlinkUI", dependencies: ["JetlinkKit"]),
    .target(name: "JetlinkONNX", dependencies: ["JetlinkLog"]),
    .target(name: "JetlinkRegistry", dependencies: ["JetlinkKit", "JetlinkLog", crypto]),
    // onnxruntime's own iOS/macOS build, the archive its CocoaPods pod and Swift
    // package ship: a static xcframework with device, simulator and macOS
    // slices, of the release Pinned names (1.29.0).
    .binaryTarget(
      name: "onnxruntime",
      url: "https://download.onnxruntime.ai/pod-archive-onnxruntime-c-1.29.0.zip",
      checksum: "ab89ea27b074201b83c12526d7f7206b916ecd5315174d372d6c27b659e49860"),
    .target(
      name: "COrt",
      dependencies: [.target(name: "onnxruntime", condition: .when(platforms: apple))],
      cSettings: [.define("JL_ORT_DLOPEN", .when(platforms: linux))],
      linkerSettings: [
        .linkedFramework("CoreML", .when(platforms: apple)),
        .linkedFramework("Foundation", .when(platforms: apple)),
        // the runtime watches the network path for its telemetry uploader
        .linkedFramework("Network", .when(platforms: apple)),
        .linkedLibrary("c++", .when(platforms: apple)),
        .linkedLibrary("dl", .when(platforms: linux)),
      ]),
    // usbdevfs's ioctls, which are macros Swift cannot import.
    .target(name: "CUsbfs"),
    .target(
      name: "JetlinkServer",
      dependencies: ["JetlinkKit", "JetlinkONNX", "JetlinkRegistry", "JetlinkLog", .target(name: "CUsbfs", condition: .when(platforms: linux)), crypto]),
    .target(
      name: "JetlinkORT", dependencies: ["JetlinkKit", "JetlinkONNX", "JetlinkServer", "COrt"],
      linkerSettings: [.linkedFramework("Metal", .when(platforms: apple))]),
    .target(
      name: "JetlinkAndroid", dependencies: ["JetlinkKit", "JetlinkServer", "JetlinkORT"],
      linkerSettings: [.linkedLibrary("log", .when(platforms: [.android]))]),
    .executableTarget(name: "jetlink-serve", dependencies: ["JetlinkKit", "JetlinkServer", "JetlinkORT"]),
    .executableTarget(name: "jetlink-onnx", dependencies: ["JetlinkONNX"]),
    // Helpers more than one test target uses: JSON comparison, hex, the source tree.
    .target(name: "JetlinkTestSupport", path: "Tests/JetlinkTestSupport"),
    .testTarget(name: "JetlinkKitTests", dependencies: ["JetlinkKit", "JetlinkUI", "JetlinkTestSupport"], resources: [.copy("Fixtures")]),
    // The fixtures are read in place through #filePath, so they are not resources.
    .testTarget(name: "JetlinkONNXTests", dependencies: ["JetlinkONNX", "JetlinkTestSupport", crypto], exclude: ["Fixtures"]),
    .testTarget(name: "JetlinkRegistryTests", dependencies: ["JetlinkRegistry", "JetlinkTestSupport", crypto]),
    // The server's tests run it on onnxruntime's CPU provider.
    .testTarget(name: "JetlinkServerTests", dependencies: ["JetlinkServer", "JetlinkORT", "JetlinkTestSupport"], exclude: ["Fixtures"]),
  ]
)
