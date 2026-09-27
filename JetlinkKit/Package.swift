// swift-tools-version: 6.2
//
// The Swift half of jetlink, shared by the Mac and iPhone apps.
//
//   JetlinkKit       the control protocol's types and the stores both apps' views read
//   JetlinkUI        SwiftUI pieces both apps draw with: the frame budget, badges, progress
//   JetlinkONNX      reading and preparing a driving model's ONNX, without the onnx package
//   JetlinkRegistry  sunnypilot's model catalog, LFS downloads, and the cache layout
//   JetlinkServer    the jetlink server in Swift: the wire protocol, the session, the
//                    queues and onnxruntime's CoreML provider. The iPhone runs it in
//                    process, the Mac app behind a setting, and `jetlink-serve` on a
//                    Mac for benches.
//
// jetlink-onnx is the preparation on its own, for checking it against Python.
//
// On Linux the package is the portable part only: no onnxruntime, CoreML, Metal, vImage or SwiftUI, and swift-crypto for
// CryptoKit. It runs the conformance suite against the Python's fixtures on a
// second platform (docs/conformance.md); nothing deploys it.
import PackageDescription

#if os(Linux)
  /// The server's files that build anywhere: the wire, the TCP and USB
  /// framing, the queues and the frame statistics.
  let portableServer = [
    "WireProtocol.swift", "FrameReader.swift", "Transport.swift", "MessageLink.swift", "USBTransport.swift", "Queues.swift", "Convert.swift",
    "ModelSpec.swift", "ElementType.swift", "FrameStats.swift", "Log.swift", "Backend.swift", "Latch.swift", "ONNXPreparer.swift",
  ]
  let crypto: Target.Dependency = .product(name: "Crypto", package: "swift-crypto")

  let package = Package(
    name: "JetlinkKit",
    products: [
      .library(name: "JetlinkKit", targets: ["JetlinkKit"]),
      .library(name: "JetlinkONNX", targets: ["JetlinkONNX"]),
      .library(name: "JetlinkRegistry", targets: ["JetlinkRegistry"]),
      .library(name: "JetlinkServer", targets: ["JetlinkServer"]),
    ],
    dependencies: [
      .package(url: "https://github.com/apple/swift-crypto.git", "3.0.0"..<"5.0.0")
    ],
    targets: [
      // os.Logger's shape, for the modules that log through it.
      .target(name: "JetlinkLog"),
      .target(name: "JetlinkKit", dependencies: ["JetlinkLog"]),
      .target(name: "JetlinkONNX", dependencies: ["JetlinkLog"]),
      .target(name: "JetlinkRegistry", dependencies: ["JetlinkKit", "JetlinkLog", crypto]),
      .target(
        name: "JetlinkServer", dependencies: ["JetlinkKit", "JetlinkONNX", "JetlinkLog"], sources: portableServer),
      .testTarget(
        name: "JetlinkKitTests", dependencies: ["JetlinkKit"], exclude: ["FormattingTests.swift"], resources: [.copy("Fixtures")]),
      .testTarget(name: "JetlinkONNXTests", dependencies: ["JetlinkONNX", crypto], exclude: ["Fixtures"]),
      .testTarget(
        name: "JetlinkRegistryTests", dependencies: ["JetlinkRegistry", crypto],
        sources: ["ConformanceTests.swift", "Support.swift", "JSONTests.swift", "CacheLayoutTests.swift"]),
      .testTarget(
        name: "JetlinkServerTests", dependencies: ["JetlinkServer"], exclude: ["Fixtures"],
        sources: ["ConformanceTests.swift", "Support.swift", "WireTests.swift", "USBTransportTests.swift", "ConvertTests.swift", "SpecTests.swift"]),
    ]
  )
#else
  let package = Package(
    name: "JetlinkKit",
    platforms: [.macOS(.v15), .iOS(.v26)],
    products: [
      .library(name: "JetlinkKit", targets: ["JetlinkKit"]),
      .library(name: "JetlinkUI", targets: ["JetlinkUI"]),
      .library(name: "JetlinkONNX", targets: ["JetlinkONNX"]),
      .library(name: "JetlinkRegistry", targets: ["JetlinkRegistry"]),
      .library(name: "JetlinkServer", targets: ["JetlinkServer"]),
      .executable(name: "jetlink-serve", targets: ["jetlink-serve"]),
      .executable(name: "jetlink-onnx", targets: ["jetlink-onnx"]),
    ],
    targets: [
      // onnxruntime's own iOS/macOS build, the archive its CocoaPods pod and Swift
      // package ship. A static xcframework with device, simulator and macOS slices.
      // The same release the Mac's Python server runs (1.29.0).
      .binaryTarget(
        name: "onnxruntime",
        url: "https://download.onnxruntime.ai/pod-archive-onnxruntime-c-1.29.0.zip",
        checksum: "ab89ea27b074201b83c12526d7f7206b916ecd5315174d372d6c27b659e49860"),
      .target(
        name: "COrt",
        dependencies: ["onnxruntime"],
        linkerSettings: [
          .linkedFramework("CoreML"),
          .linkedFramework("Foundation"),
          // the runtime watches the network path for its telemetry uploader
          .linkedFramework("Network"),
          .linkedLibrary("c++"),
        ]),
      .target(name: "JetlinkKit"),
      .target(name: "JetlinkUI", dependencies: ["JetlinkKit"]),
      .target(name: "JetlinkONNX"),
      .target(name: "JetlinkRegistry", dependencies: ["JetlinkKit"]),
      .target(
        name: "JetlinkServer",
        dependencies: ["JetlinkKit", "JetlinkONNX", "JetlinkRegistry", "COrt"],
        linkerSettings: [.linkedFramework("Metal")]),
      .executableTarget(name: "jetlink-serve", dependencies: ["JetlinkKit", "JetlinkServer"]),
      .executableTarget(name: "jetlink-onnx", dependencies: ["JetlinkONNX"]),
      .testTarget(name: "JetlinkKitTests", dependencies: ["JetlinkKit", "JetlinkUI"], resources: [.copy("Fixtures")]),
      // The fixtures are read in place through #filePath, so they are not resources.
      .testTarget(name: "JetlinkONNXTests", dependencies: ["JetlinkONNX"], exclude: ["Fixtures"]),
      .testTarget(name: "JetlinkRegistryTests", dependencies: ["JetlinkRegistry"]),
      .testTarget(name: "JetlinkServerTests", dependencies: ["JetlinkServer"], exclude: ["Fixtures"]),
    ]
  )
#endif
