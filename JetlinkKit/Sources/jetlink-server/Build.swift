#if os(macOS) || os(Linux)
  import ArgumentParser
  import Foundation
  import JetlinkKit
  import JetlinkRegistry
  import JetlinkServer

  /// `jetlink-server build ONNX`: the engine a comma would get, built ahead
  /// of the drive (Python's --build).
  struct Build: ParsableCommand {
    static let configuration = CommandConfiguration(
      abstract: "Build this backend's engine for a model, or say it is built already.",
      discussion: "Stop a running server on the same cache first: both would build into it.")

    @Argument(help: "The model's ONNX.")
    var onnx: String
    @Option(help: "Model frames per camera frame the comma sends; the spec in the sidecar and the preload use it.")
    var frameSkip = Pinned.defaultFrameSkip
    @OptionGroup var chosen: BackendArguments
    @OptionGroup var cache: CacheArguments
    @Option(help: "debug, info, warning or error.")
    var logLevel = LogLevel.info

    func validate() throws {
      guard frameSkip > 0 else { throw ValidationError("--frame-skip must be at least 1") }
    }

    func run() throws {
      setUpLogging(logLevel)
      try buildEngine(model: URL(fileURLWithPath: onnx), frameSkip: frameSkip, backend: chosen.pick(), root: cache.root)
    }
  }

  /// Builds the engine for `model` into the cache at `root`, unless it is
  /// there. Logs each stage every 2 %, carries the spec into the sidecar as
  /// a served build does, so the first comma loads rather than reparses, and
  /// with nothing recorded as loaded last, records this model so the next
  /// server start preloads it. Throws ExitCode.failure after logging why.
  func buildEngine(model: URL, frameSkip: Int, backend: any EngineBackend, root: URL) throws {
    let log = ServerLog(category: "main")
    do {
      log.info("backend \(backend.name) \(backend.runtimeVersion) on \(backend.deviceTag()), cache \(root.path)")
      let cache = try ServerCache(root: root, backend: backend)
      let (sha256, nbytes) = try Registry.hashFile(model)
      let entry = try cache.entry(sha256)
      log.info("model \(sha256.prefix(16)) (\(nbytes >> 20) MB) -> \(entry.path.path)")
      if entry.exists {
        log.info("already built: \(sidecar(entry))")
        return
      }
      let spec = try backend.deriveSpec(model: model, sha256: sha256, nbytes: nbytes, frameSkip: frameSkip)
      try backend.build(model: model, artifact: entry.path, report: stageLog { log.info($0) }, metaExtra: ["spec": spec.dictionary()])
      log.info("built: \(sidecar(entry))")
      if cache.lastLoaded() == nil {
        cache.rememberLoaded(sha256, frameSkip: spec.frameSkip)
      }
    } catch {
      log.error("could not build \(model.path): \(error)")
      throw ExitCode.failure
    }
  }

  /// A build's or a load's progress as log lines: each stage's start, every
  /// 2 % of it, and its end. Per stage, since each runs 0 to 1, and a
  /// fraction carried over from the last would hide the next one's lines.
  func stageLog(_ write: @escaping @Sendable (String) -> Void) -> ProgressFn {
    let seen = Locked<(stage: String?, frac: Double)>((nil, 0))
    return { stage, frac, msg in
      let due = seen.withLock { seen in
        if stage != seen.stage { seen = (stage, 0) }
        guard frac - seen.frac >= 0.02 || frac >= 1 || frac == 0 else { return false }
        seen.frac = frac
        return true
      }
      if due {
        write("\(stage.padding(toLength: max(8, stage.count), withPad: " ", startingAt: 0)) \(String(format: "%5.1f", frac * 100))%  \(msg)")
      }
    }
  }

  /// The sidecar on one line, as the journal keeps a line.
  private func sidecar(_ entry: CacheEntry) -> String {
    (try? Data(contentsOf: entry.metaPath)).flatMap { try? JSONSerialization.jsonObject(with: $0) }.flatMap { try? jsonText($0) } ?? "(no sidecar)"
  }

  /// `jetlink-server spec ONNX`: the ModelSpec JSON the comma is sent for a
  /// model (Python's --dump-spec).
  struct Spec: ParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Print the spec the comma is sent for a model, as JSON.")

    @Argument(help: "The model's ONNX.")
    var onnx: String

    func run() throws {
      let model = URL(fileURLWithPath: onnx)
      let (sha256, nbytes) = try Registry.hashFile(model)
      let spec = try ONNXPreparer().readSpec(model: model, sha256: sha256, nbytes: nbytes, frameSkip: Pinned.defaultFrameSkip)
      print(try jsonText(spec.dictionary()))
    }
  }

  /// One line of JSON, keys sorted so two runs print the same.
  func jsonText(_ object: Any) throws -> String {
    String(decoding: try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys, .withoutEscapingSlashes]), as: UTF8.self)
  }

  /// `jetlink-server bench`: a built engine at the comma's pace, 20 frames a
  /// second through the real queues, with no comma and no link, so the
  /// server's own share of a frame can be read at home.
  struct Bench: ParsableCommand {
    static let configuration = CommandConfiguration(
      abstract: "Run a built engine at the comma's pace with no comma, and report its times.",
      discussion: "Stop a running server first: the engine needs the device to itself.")

    @Option(help: "How long to run.")
    var seconds = 60.0
    @Option(help: "The model to run. Default: the one loaded last.")
    var sha256: String?
    @Option(help: "Model frames per camera frame. Default: what it was loaded with last, else 4.")
    var frameSkip: Int?
    @OptionGroup var chosen: BackendArguments
    @OptionGroup var cache: CacheArguments
    @Option(help: "debug, info, warning or error.")
    var logLevel = LogLevel.info

    func validate() throws {
      guard seconds > 0 && seconds <= 3600 else { throw ValidationError("--seconds must be above 0 and at most 3600") }
      if let sha256, !CacheLayout.isSHA256(sha256) { throw ValidationError("--sha256 wants a 64 character sha256, not \(sha256)") }
      if let frameSkip, frameSkip < 1 { throw ValidationError("--frame-skip must be at least 1") }
    }

    func run() throws {
      setUpLogging(logLevel)
      let log = ServerLog(category: "main")
      let backend = try chosen.pick()
      do {
        let server = try Server(
          configuration: Server.Configuration(cacheRoot: cache.root, preload: false, listen: false), backend: backend)
        defer { server.shutdown() }
        let last = server.cache.lastLoaded()
        guard let wanted = sha256 ?? last?.sha256 else {
          throw HostError.failed("no model has been loaded here: name one with --sha256")
        }
        guard try server.cache.entry(wanted).exists else {
          throw HostError.failed("\(wanted.prefix(16)) is not built for \(backend.tag()): build it first")
        }
        try load(wanted, frameSkip: frameSkip ?? (last?.sha256 == wanted ? last!.frameSkip : Pinned.defaultFrameSkip), on: server)
        // The report goes to stdout alone, where a script reads it.
        let report = try server.host.benchmark(seconds: seconds, run: BenchmarkRun(), logsReport: false)
        print(report.text)
      } catch let exit as ExitCode {
        throw exit
      } catch {
        log.error("\(error)")
        throw ExitCode.failure
      }
    }

    /// Loads the model through the host's own request, the one piece of
    /// code that loads an engine, and waits for it.
    private func load(_ sha256: String, frameSkip: Int, on server: Server) throws {
      let done = DispatchSemaphore(value: 0)
      let log = ServerLog(category: "main")
      let progress = stageLog { log.info($0) }
      server.host.subscribe { event in
        switch event {
        case .progress(let stage, let frac, let msg): progress(stage, frac, msg)
        case .engine(let engine) where engine.sha256 == sha256 && (engine.state == .ready || engine.state == .failed): done.signal()
        default: break
        }
      }
      let controller = ServerController(server: server, registry: Registry(layout: server.cache.layout))
      let reply = try blocking { await controller.handle(.prepare(sha256: sha256, frameSkip: frameSkip)) }
      guard reply.ok else { throw HostError.failed(reply.error ?? "could not load \(sha256.prefix(16))") }
      let started = server.host.snapshot()
      guard started.sha256 == sha256, started.state != .none else {
        throw HostError.failed("could not load \(sha256.prefix(16)): its sidecar has no spec and the model is not here")
      }
      done.wait()
      let engine = server.host.snapshot()
      guard engine.state == .ready, engine.sha256 == sha256 else {
        throw HostError.failed("could not load \(sha256.prefix(16)): \(engine.detail)")
      }
    }
  }

  /// Waits on the calling thread for async work: a command runs start to end
  /// on the main thread, which the cooperative pool does not need.
  func blocking<T: Sendable>(_ work: @escaping @Sendable () async throws -> T) throws -> T {
    let result = Locked<Result<T, any Error>?>(nil)
    let done = DispatchSemaphore(value: 0)
    Task.detached {
      let outcome: Result<T, any Error>
      do {
        outcome = .success(try await work())
      } catch {
        outcome = .failure(error)
      }
      result.withLock { $0 = outcome }
      done.signal()
    }
    done.wait()
    return try result.withLock { $0! }.get()
  }

  /// A value behind a lock that escaping closures can share, which Mutex,
  /// being noncopyable, cannot be.
  final class Locked<Value>: @unchecked Sendable {
    private let lock = NSLock()
    private var value: Value

    init(_ value: Value) {
      self.value = value
    }

    func withLock<R>(_ body: (inout Value) throws -> R) rethrows -> R {
      lock.lock()
      defer { lock.unlock() }
      return try body(&value)
    }
  }
#endif
