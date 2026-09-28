#if os(macOS) || os(Linux)
  import ArgumentParser
  import Foundation
  import JetlinkKit
  import JetlinkRegistry
  import JetlinkServer
  import Testing

  @testable import jetlink_server

  struct ParsingTests {
    private func root(_ arguments: [String]) throws -> any ParsableCommand {
      try JetlinkServerCommand.parseAsRoot(arguments)
    }

    @Test func serveIsTheDefaultAndTakesWhatTheInstalledUnitPasses() throws {
      let command = try root(["--usb", "--backend", "trt", "--cache", "/var/lib/jetlink", "--sleep-after", "900", "--status-port", "5600"])
      let serve = try #require(command as? Serve)
      #expect(serve.usb && !serve.listen)
      #expect(serve.chosen.backend == .trt && serve.chosen.device == nil)
      #expect(serve.cache.root.path == "/var/lib/jetlink")
      #expect(serve.sleepAfter == 900 && serve.statusPort == 5600)
    }

    @Test func serveDefaults() throws {
      let serve = try #require(try root([]) as? Serve)
      #expect(!serve.usb && !serve.listen && serve.dial == nil)
      #expect(serve.host == "0.0.0.0" && serve.port == 5599 && serve.statusPort == 0 && serve.sleepAfter == 0)
      #expect(serve.chosen.backend == .auto && serve.logLevel == .info)
      #expect(!serve.noPreload && !serve.noKeepAlive && !serve.noCPUKeepWarm)
      let spelled = try #require(try root(["serve", "--listen", "--dial", "192.168.60.1:5599", "--log-level", "debug"]) as? Serve)
      #expect(spelled.listen && spelled.dial == "192.168.60.1:5599" && spelled.logLevel == .debug)
    }

    @Test func usageMistakesAreRefused() {
      for arguments in [
        ["--sleep-after", "900"],  // without --usb
        ["--sleep-after", "-1", "--usb"],
        ["--dial", "a:b:c"],
        ["--status-port", "70000"],
        ["--backend", "cuda"],
        ["--bogus"],
        ["build", "m.onnx", "--frame-skip", "0"],
        ["bench", "--seconds", "0"],
        ["bench", "--sha256", "nothex"],
        ["models", "fetch"],
      ] {
        #expect(throws: (any Error).self, "\(arguments)") { try root(arguments) }
      }
    }

    /// runAndExit turns exactly these into exit 1, as every other error is.
    @Test func aUsageMistakeIsAValidationFailure() {
      do {
        _ = try root(["--bogus"])
        Issue.record("--bogus parsed")
      } catch {
        #expect(JetlinkServerCommand.exitCode(for: error) == .validationFailure)
      }
    }

    @Test func theOtherCommands() throws {
      let build = try #require(try root(["build", "m.onnx", "--frame-skip", "2", "--backend", "ort", "--device", "cpu"]) as? Build)
      #expect(build.onnx == "m.onnx" && build.frameSkip == 2 && build.chosen.options().device == "cpu")
      #expect(try root(["spec", "m.onnx"]) is Spec)
      let backends = try #require(try root(["backends", "--backend", "trt"]) as? ListBackends)
      #expect(backends.chosen.backend == .trt)
      let bench = try #require(try root(["bench", "--seconds", "5", "--sha256", String(repeating: "a", count: 64)]) as? Bench)
      #expect(bench.seconds == 5 && bench.frameSkip == nil)
    }

    @Test func modelsSubcommands() throws {
      let sha = String(repeating: "b", count: 64)
      let rm = try #require(try root(["models", "rm", sha, "--artifacts", "--model", "--cache", "/c"]) as? Models.Remove)
      #expect(rm.sha256 == sha && rm.artifacts && rm.model && rm.cache.root.path == "/c")
      let prepare = try #require(try root(["models", "prepare", RegistryFixture.ref, "--backend", "auto", "--device", "auto"]) as? Models.Prepare)
      #expect(prepare.chosen.options().device == nil, "auto is each backend's default, as Python's --device auto was")
      #expect(try root(["models", "list", "--refresh", "--json"]) is Models.List)
      #expect(try root(["models", "inventory", "--json"]) is Models.Inventory)
      let imported = try #require(try root(["models", "import", "/m.onnx", "--name", "mine"]) as? Models.Import)
      #expect(imported.name == "mine")
    }
  }

  struct VersionTests {
    @Test func aVersionFileBesideBinWins() throws {
      let tmp = try TempDir()
      defer { tmp.remove() }
      let binary = tmp.url.appending(path: "bin/jetlink-server")
      #expect(productVersion(executable: binary) == Pinned.productVersion)
      try "0.7.0-dev.8b8268e\n".write(to: tmp.url.appending(path: "VERSION"), atomically: true, encoding: .utf8)
      #expect(productVersion(executable: binary) == "0.7.0-dev.8b8268e")
      try "\n".write(to: tmp.url.appending(path: "VERSION"), atomically: true, encoding: .utf8)
      #expect(productVersion(executable: binary) == Pinned.productVersion)
      #expect(productVersion(executable: nil) == Pinned.productVersion)
    }
  }

  struct ShutdownTests {
    @Test func theServerStopsBeforeThePage() {
      let order = Locked<[String]>([])
      shutDown(
        [("the server", { order.withLock { $0.append("server") } }), ("the status page", { order.withLock { $0.append("page") } })],
        log: ServerLog(category: "test"))
      #expect(order.withLock { $0 } == ["server", "page"])
    }
  }

  struct CacheDefaultTests {
    @Test func jetlinkCacheWins() {
      #expect(defaultCache(environment: ["JETLINK_CACHE": "/somewhere"], tegra: { true }).path == "/somewhere")
    }

    @Test func aTegraUsesItsDataPartition() {
      #expect(defaultCache(environment: [:], tegra: { true }).path == "/mnt/data/jetlink")
    }

    @Test(.enabled(if: !FileManager.default.fileExists(atPath: "/mnt/data/jetlink")))
    func elseTheUsersCacheDirectory() {
      let home = FileManager.default.homeDirectoryForCurrentUser.path
      #if os(macOS)
        #expect(defaultCache(environment: [:], tegra: { false }).path == home + "/Library/Caches/jetlink")
      #else
        #expect(defaultCache(environment: [:], tegra: { false }).path == home + "/.cache/jetlink")
        #expect(defaultCache(environment: ["XDG_CACHE_HOME": "/xdg"], tegra: { false }).path == "/xdg/jetlink")
      #endif
    }
  }

  struct BuildTests {
    /// onnxruntime's CPU provider, which every platform's tests have.
    private func cpu() throws -> any EngineBackend {
      try BackendOptions(device: "cpu").ort()
    }

    @Test func buildsOnceCarriesTheSpecAndRecordsThePreload() throws {
      let tmp = try TempDir()
      defer { tmp.remove() }
      let backend = try cpu()
      try buildEngine(model: Tiny.queued, frameSkip: 2, backend: backend, root: tmp.url)
      let cache = try ServerCache(root: tmp.url, backend: backend)
      let entry = try cache.entry(try Tiny.sha256(Tiny.queued))
      #expect(entry.exists)
      let meta = try #require(try JSONSerialization.jsonObject(with: Data(contentsOf: entry.metaPath)) as? [String: Any])
      let spec = try #require(meta["spec"] as? [String: Any])
      #expect(spec["sha256"] as? String == (try Tiny.sha256(Tiny.queued)) && (spec["frame_skip"] as? NSNumber)?.intValue == 2)
      #expect(cache.lastLoaded()?.sha256 == (try Tiny.sha256(Tiny.queued)) && cache.lastLoaded()?.frameSkip == 2)

      // Built already: nothing to do. A second model leaves the preload alone.
      let built = try FileManager.default.attributesOfItem(atPath: entry.metaPath.path)[.modificationDate] as? Date
      try buildEngine(model: Tiny.queued, frameSkip: 2, backend: backend, root: tmp.url)
      #expect(try FileManager.default.attributesOfItem(atPath: entry.metaPath.path)[.modificationDate] as? Date == built)
      try buildEngine(model: Tiny.stateful, frameSkip: 4, backend: backend, root: tmp.url)
      #expect(try cache.entry(try Tiny.sha256(Tiny.stateful)).exists)
      #expect(cache.lastLoaded()?.sha256 == (try Tiny.sha256(Tiny.queued)))
    }

    @Test func aModelThatIsNotThereFails() throws {
      let tmp = try TempDir()
      defer { tmp.remove() }
      #expect(throws: ExitCode.failure) {
        try buildEngine(model: tmp.url.appending(path: "missing.onnx"), frameSkip: 4, backend: try cpu(), root: tmp.url)
      }
    }

    @Test func theStageLogKeepsEveryTwoPercentOfEachStage() {
      let lines = Locked<[String]>([])
      let report = stageLog { line in lines.withLock { $0.append(line) } }
      for (stage, frac) in [("build", 0.0), ("build", 0.01), ("build", 0.021), ("build", 0.03), ("build", 1.0), ("load", 0.5), ("load", 0.51)] {
        report(stage, frac, "working")
      }
      #expect(
        lines.withLock { $0 } == [
          "build      0.0%  working", "build      2.1%  working", "build    100.0%  working", "load      50.0%  working",
        ])
    }
  }
#endif
