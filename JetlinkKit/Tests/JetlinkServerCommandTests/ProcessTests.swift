#if os(macOS) || os(Linux)
  import Foundation
  import JetlinkKit
  import Testing

  #if canImport(Glibc)
    import Glibc
  #endif

  /// The built binary beside this test bundle, or $JETLINK_SERVER_BIN.
  enum Binary {
    static let url: URL? = {
      if let named = ProcessInfo.processInfo.environment["JETLINK_SERVER_BIN"], !named.isEmpty {
        return URL(fileURLWithPath: named)
      }
      #if os(Linux)
        let directory = (try? FileManager.default.destinationOfSymbolicLink(atPath: "/proc/self/exe")).map {
          URL(fileURLWithPath: $0).deletingLastPathComponent()
        }
      #else
        let directory = Optional(Bundle(for: Marker.self).bundleURL.deletingLastPathComponent())
      #endif
      let url = directory?.appending(path: "jetlink-server")
      return url.flatMap { FileManager.default.isExecutableFile(atPath: $0.path) ? $0 : nil }
    }()

    private final class Marker {}

    /// Runs it to the end: (status, stdout, stderr).
    static func run(_ arguments: [String], binary: URL? = url, environment: [String: String] = [:]) throws -> (Int32, String, String) {
      let process = Process()
      process.executableURL = try #require(binary)
      process.arguments = arguments
      process.environment = ProcessInfo.processInfo.environment.merging(environment) { _, new in new }
      let (out, err) = (Pipe(), Pipe())
      process.standardOutput = out
      process.standardError = err
      try process.run()
      // Read before waiting: a full pipe would stall the child.
      let output = out.fileHandleForReading.readDataToEndOfFile()
      let errors = err.fileHandleForReading.readDataToEndOfFile()
      process.waitUntilExit()
      return (process.terminationStatus, String(decoding: output, as: UTF8.self), String(decoding: errors, as: UTF8.self))
    }
  }

  @Suite(.enabled(if: Binary.url != nil, "the jetlink-server product is built"))
  struct ProcessTests {
    @Test func versionIsOneLine() throws {
      let (status, out, _) = try Binary.run(["--version"])
      #expect(status == 0 && out == Pinned.productVersion + "\n")
    }

    /// The release tarball's layout: bin/jetlink-server beside VERSION.
    @Test func versionReadsTheTarballsVersionFile() throws {
      let binary = try #require(Binary.url)
      // Beside the build, so a hard link needs no copy of the binary.
      let root = binary.deletingLastPathComponent().deletingLastPathComponent()
        .appending(path: "version-test-\(UUID().uuidString)", directoryHint: .isDirectory)
      defer { try? FileManager.default.removeItem(at: root) }
      let linked = root.appending(path: "bin/jetlink-server")
      try FileManager.default.createDirectory(at: linked.deletingLastPathComponent(), withIntermediateDirectories: true)
      do {
        try FileManager.default.linkItem(at: binary, to: linked)
      } catch {
        try FileManager.default.copyItem(at: binary, to: linked)
      }
      try "0.7.0-dev.8b8268e\n".write(to: root.appending(path: "VERSION"), atomically: true, encoding: .utf8)
      let (status, out, _) = try Binary.run(["--version"], binary: linked)
      #expect(status == 0 && out == "0.7.0-dev.8b8268e\n")
    }

    @Test func usageMistakesExitOne() throws {
      let (status, _, err) = try Binary.run(["--bogus"])
      #expect(status == 1 && err.contains("--bogus"))
      #expect(try Binary.run(["models", "rm", "nothex", "--model", "--cache", NSTemporaryDirectory()]).0 == 1)
    }

    @Test func specIsWhatPythonDerives() throws {
      let (status, out, err) = try Binary.run(["spec", Tiny.queued.path])
      #expect(status == 0, "\(err)")
      let printed = try #require(try JSONSerialization.jsonObject(with: Data(out.utf8)) as? NSDictionary)
      #expect(printed == (try Tiny.spec(Tiny.queued)) as NSDictionary)
    }

    /// The installer's GPU check: TensorRT or a non-zero exit.
    @Test func backendsSaysWhyAndExitsOnTheOneAskedFor() throws {
      let (status, out, _) = try Binary.run(["backends"])
      #expect(out.contains("trt: ") && out.contains("ort: "))
      #expect(status == (out.contains("usable: ") ? 0 : 1))
      let (trt, said, _) = try Binary.run(["backends", "--backend", "trt"])
      #expect(trt == (said.contains("trt: usable") ? 0 : 1))
    }

    @Test func buildThenBench() throws {
      let tmp = try TempDir()
      defer { tmp.remove() }
      let cpu = ["--backend", "ort", "--device", "cpu", "--cache", tmp.path]
      let (built, _, buildLog) = try Binary.run(["build", Tiny.stateful.path] + cpu)
      #expect(built == 0, "\(buildLog)")
      #expect(buildLog.contains("built: {"))
      let (again, _, againLog) = try Binary.run(["build", Tiny.stateful.path] + cpu)
      #expect(again == 0 && againLog.contains("already built: {"))
      let (benched, report, benchLog) = try Binary.run(["bench", "--seconds", "1"] + cpu)
      #expect(benched == 0, "\(benchLog)")
      #expect(report.hasPrefix("Jetlink benchmark") && report.contains("over 35 ms: "), "\(report)")
      #expect(!benchLog.contains("Jetlink benchmark"), "the report is on stdout only")
    }

    /// SIGTERM stops it cleanly, in order: the server, then the status page.
    @Test func sigtermStopsItCleanly() throws {
      let tmp = try TempDir()
      defer { tmp.remove() }
      let port = try freePort()
      var page = try freePort()
      while page == port { page = try freePort() }
      let log = tmp.url.appending(path: "server.log")
      FileManager.default.createFile(atPath: log.path, contents: nil)
      let handle = try FileHandle(forWritingTo: log)
      let process = Process()
      process.executableURL = try #require(Binary.url)
      process.arguments = [
        "--backend", "ort", "--device", "cpu", "--listen", "--host", "127.0.0.1", "--port", "\(port)", "--cache", tmp.path, "--no-preload",
        "--status-port", "\(page)",
      ]
      process.standardOutput = handle
      process.standardError = handle
      try process.run()
      defer { if process.isRunning { process.terminate() } }

      let deadline = Date().addingTimeInterval(30)
      while !canConnect(port) || !canConnect(page) {
        guard process.isRunning, Date() < deadline else {
          Issue.record("never listened: \(String(decoding: (try? Data(contentsOf: log)) ?? Data(), as: UTF8.self))")
          return
        }
        Thread.sleep(forTimeInterval: 0.1)
      }
      kill(process.processIdentifier, SIGTERM)
      let stopped = Date().addingTimeInterval(10)
      while process.isRunning && Date() < stopped { Thread.sleep(forTimeInterval: 0.05) }
      #expect(!process.isRunning)
      #expect(process.terminationReason == .exit && process.terminationStatus == 0)
      let text = String(decoding: try Data(contentsOf: log), as: UTF8.self)
      let stopping = try #require(text.range(of: "stopping on SIGTERM"), "\(text)")
      let server = try #require(text.range(of: "stopped the server"), "\(text)")
      let statusPage = try #require(text.range(of: "stopped the status page"), "\(text)")
      #expect(stopping.upperBound <= server.lowerBound)
      #expect(server.upperBound <= statusPage.lowerBound)
    }

    private func freePort() throws -> UInt16 {
      let listener = try #require(Optional(socket(AF_INET, socketStream, 0)).flatMap { $0 >= 0 ? $0 : nil })
      defer { close(listener) }
      var address = loopback(0)
      let bound = withUnsafePointer(to: &address) {
        $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(listener, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) }
      }
      try #require(bound == 0)
      var length = socklen_t(MemoryLayout<sockaddr_in>.size)
      _ = withUnsafeMutablePointer(to: &address) {
        $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { getsockname(listener, $0, &length) }
      }
      return UInt16(bigEndian: address.sin_port)
    }

    private func canConnect(_ port: UInt16) -> Bool {
      let fd = socket(AF_INET, socketStream, 0)
      guard fd >= 0 else { return false }
      defer { close(fd) }
      var address = loopback(port)
      return withUnsafePointer(to: &address) {
        $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { connect(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) }
      } == 0
    }

    private func loopback(_ port: UInt16) -> sockaddr_in {
      var address = sockaddr_in()
      address.sin_family = sa_family_t(AF_INET)
      address.sin_addr.s_addr = inet_addr("127.0.0.1")
      address.sin_port = port.bigEndian
      return address
    }

    #if os(Linux)
      private let socketStream = Int32(SOCK_STREAM.rawValue)
    #else
      private let socketStream = SOCK_STREAM
    #endif
  }
#endif
