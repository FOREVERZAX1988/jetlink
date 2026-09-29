// swift-tools-version: 6.2
//
// The Swift half of jetlink: the server, and what the apps share. One manifest
// for every platform: what differs is conditioned on the platform being built
// for, and a file for one platform compiles to nothing on the others, so every
// target builds everywhere and every test suite runs wherever it builds.
//
// onnxruntime is linked in on Apple platforms and opened at run time elsewhere
// (JL_ORT_DLOPEN): the app's copy from the AAR on Android, the official
// tarball's on Linux. Building for either needs onnxruntime's C headers, with
// -Xcc -I<a directory holding onnxruntime/onnxruntime_c_api.h>;
// android/scripts/swift-build.sh passes the AAR's.
//
// CTrt is the real shim over TensorRT only on Linux with JETLINK_TENSORRT set to
// a directory holding TensorRT's and CUDA's headers (scripts/build-linux.sh
// fetches them). Everywhere else it is the fake over host memory, which the
// tests run on and which never opens TensorRT, so a build without the headers
// cannot serve with it.
import PackageDescription

let apple: [Platform] = [.macOS, .iOS]
/// No CryptoKit and no onnxruntime framework: swift-crypto, and dlopen.
let linux: [Platform] = [.linux, .android]

#if os(Linux)
  let tensorRT = Context.environment["JETLINK_TENSORRT"]
#else
  let tensorRT: String? = nil
#endif

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
    .executable(name: "jetlink-server", targets: ["jetlink-server"]),
  ],
  dependencies: [
    .package(url: "https://github.com/apple/swift-crypto.git", "3.0.0"..<"5.0.0"),
    .package(url: "https://github.com/apple/swift-argument-parser.git", exact: "1.8.2"),
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
      dependencies: ["JetlinkKit", "JetlinkONNX", "JetlinkRegistry", "JetlinkLog", .target(name: "CUsbfs", condition: .when(platforms: linux))]),
    .target(
      name: "JetlinkORT", dependencies: ["JetlinkKit", "JetlinkONNX", "JetlinkServer", "COrt"],
      linkerSettings: [.linkedFramework("Metal", .when(platforms: apple))]),
    .target(
      name: "CTrt",
      exclude: [tensorRT == nil ? "jl_trt.cpp" : "jl_trt_fake.c"],
      cxxSettings: tensorRT.map { [.unsafeFlags(["-isystem", $0])] } ?? [],
      linkerSettings: [.linkedLibrary("dl", .when(platforms: [.linux])), .linkedLibrary("m", .when(platforms: [.linux]))]),
    .target(name: "JetlinkTRT", dependencies: ["CTrt", "JetlinkKit", "JetlinkServer", "JetlinkONNX"]),
    .target(
      name: "JetlinkLinux",
      dependencies: [
        "JetlinkKit", "JetlinkLog", "JetlinkServer", "JetlinkStatusPage", .target(name: "CUsbfs", condition: .when(platforms: linux)),
      ]),
    .target(name: "JetlinkStatusPage", dependencies: ["JetlinkKit", "JetlinkLog", "JetlinkServer"], resources: [.copy("Resources")]),
    .target(
      name: "JetlinkAndroid", dependencies: ["JetlinkKit", "JetlinkServer", "JetlinkORT"],
      linkerSettings: [.linkedLibrary("log", .when(platforms: [.android]))]),
    // Built for Linux and macOS; elsewhere its sources compile to an empty program.
    .executableTarget(
      name: "jetlink-server",
      dependencies: [
        "JetlinkKit", "JetlinkLog", "JetlinkRegistry", "JetlinkServer", "JetlinkORT", "JetlinkStatusPage",
        .target(name: "JetlinkTRT", condition: .when(platforms: [.linux])),
        .target(name: "JetlinkLinux", condition: .when(platforms: [.linux])),
        .product(name: "ArgumentParser", package: "swift-argument-parser", condition: .when(platforms: [.macOS, .linux])),
      ]),
    // Helpers more than one test target uses: fixtures, scratch directories, waits, a fake network.
    .target(name: "JetlinkTestSupport", dependencies: ["JetlinkRegistry", "JetlinkServer"], path: "Tests/JetlinkTestSupport"),
    .testTarget(name: "JetlinkKitTests", dependencies: ["JetlinkKit", "JetlinkUI", "JetlinkTestSupport"], resources: [.copy("Fixtures")]),
    // The fixtures are read in place through #filePath, so they are not resources.
    .testTarget(name: "JetlinkONNXTests", dependencies: ["JetlinkONNX", "JetlinkTestSupport", crypto], exclude: ["Fixtures"]),
    .testTarget(name: "JetlinkRegistryTests", dependencies: ["JetlinkRegistry", "JetlinkTestSupport", crypto]),
    // The server's tests run it on onnxruntime's CPU provider.
    .testTarget(name: "JetlinkServerTests", dependencies: ["JetlinkServer", "JetlinkORT", "JetlinkTestSupport"], exclude: ["Fixtures"]),
    // On the fake shim (JL_TRT_FAKE), which jl_trt_fake.h drives.
    .testTarget(
      name: "JetlinkTRTTests", dependencies: ["JetlinkTRT", "CTrt", "JetlinkServer", "JetlinkONNX", "JetlinkTestSupport"],
      swiftSettings: tensorRT == nil ? [.define("JL_TRT_FAKE")] : []),
    // Captured sysfs trees, read in place.
    .testTarget(
      name: "JetlinkLinuxTests",
      dependencies: [
        "JetlinkLinux", "JetlinkServer", "JetlinkStatusPage", "JetlinkTestSupport", .target(name: "CUsbfs", condition: .when(platforms: linux)),
      ],
      exclude: ["Fixtures"]),
    .testTarget(name: "JetlinkStatusPageTests", dependencies: ["JetlinkStatusPage", "JetlinkKit", "JetlinkServer", "JetlinkTestSupport"]),
    // jetlink-server's commands in process, and the built binary for what
    // only a process shows (--version beside a VERSION file, SIGTERM).
    .testTarget(
      name: "JetlinkServerCommandTests",
      dependencies: [
        .target(name: "jetlink-server", condition: .when(platforms: [.macOS, .linux])),
        "JetlinkKit", "JetlinkRegistry", "JetlinkServer", "JetlinkTestSupport",
        .product(name: "ArgumentParser", package: "swift-argument-parser", condition: .when(platforms: [.macOS, .linux])),
      ]),
  ],
  cxxLanguageStandard: .cxx17
)
