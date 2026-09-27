import CryptoKit
import Foundation
import JetlinkKit

/// An engine resident on the device, with the state that goes with it.
final class Loaded {
  let sha256: String
  let spec: ModelSpec
  let engine: any Engine
  let staging: any FrameStaging

  init(sha256: String, spec: ModelSpec, engine: any Engine, staging: any FrameStaging) {
    self.sha256 = sha256
    self.spec = spec
    self.engine = engine
    self.staging = staging
  }
}

/// One build or load in flight, or its outcome. Guarded by the host's lock.
final class Job: @unchecked Sendable {
  enum State: String { case building, ready, failed }

  let sha256: String
  let loadOnly: Bool
  var state: State = .building
  var detail = ""

  init(sha256: String, loadOnly: Bool) {
    self.sha256 = sha256
    self.loadOnly = loadOnly
  }
}

/// What a client asked for: enough to identify the model without the file.
struct Request: Equatable {
  let sha256: String
  let nbytes: Int64
  let frameSkip: Int

  init(sha256: String, nbytes: Int64, frameSkip: Int) throws {
    guard EngineCache.isSHA256(sha256) else { throw HostError.invalid("model identity must be a lowercase SHA-256 digest") }
    guard nbytes >= 0, frameSkip > 0 else { throw HostError.invalid("invalid model size or frame skip") }
    self.sha256 = sha256
    self.nbytes = nbytes
    self.frameSkip = frameSkip
  }
}

enum HostError: Error, CustomStringConvertible {
  case invalid(String)
  case failed(String)

  var description: String {
    switch self {
    case .invalid(let detail), .failed(let detail): return detail
    }
  }
}

/// What the host tells its listeners: the control layer, and through it the app.
public enum HostEvent: Sendable {
  case progress(stage: String, frac: Double, msg: String)
  case engine(EngineEvent)
  case link(LinkEvent)
  /// Once a second while a comma is connected and sending.
  case stats(StatsEvent)
  /// A benchmark's progress and its report.
  case benchmark(BenchmarkEvent)
  /// The comma asked for a power-off, which this server refused.
  case shutdownRequested(reason: String)
}

/// A rolling window of served frames, for the stats event once a second.
final class FrameStats: @unchecked Sendable {
  struct Sample {
    let at: TimeInterval
    let totalUs: UInt32
    let gpuUs: UInt32
    let queueUs: UInt32
    let sendUs: UInt32
  }

  /// Frames over this count as slow, as the Python server counts them.
  static let slowUs: UInt32 = 60_000

  private let lock = NSLock()
  private var samples: [Sample] = []
  private let capacity = 2000

  func record(totalUs: UInt32, gpuUs: UInt32, queueUs: UInt32, sendUs: UInt32) {
    let sample = Sample(at: ProcessInfo.processInfo.systemUptime, totalUs: totalUs, gpuUs: gpuUs, queueUs: queueUs, sendUs: sendUs)
    lock.lock()
    samples.append(sample)
    // Trimmed in chunks: one removeFirst a frame would shift the window every frame.
    if samples.count > capacity + capacity / 4 {
      samples.removeFirst(samples.count - capacity)
    }
    lock.unlock()
  }

  /// The `stats` event, or nil when no frame landed in the window.
  func summary(window seconds: TimeInterval, framesTotal: Int) -> StatsEvent? {
    let cutoff = ProcessInfo.processInfo.systemUptime - seconds
    lock.lock()
    let rows = samples.filter { $0.at >= cutoff }
    lock.unlock()
    guard !rows.isEmpty else { return nil }
    let n = Double(rows.count)
    func mean(_ value: (Sample) -> UInt32) -> Double {
      rows.reduce(0.0) { $0 + Double(value($1)) } / n / 1000
    }
    func spread(_ values: [UInt32]) -> StatsEvent.Total {
      let sorted = values.sorted()
      let meanValue = sorted.reduce(0.0) { $0 + Double($1) } / Double(sorted.count) / 1000
      let p99 = Double(sorted[Int(0.99 * Double(sorted.count - 1))]) / 1000
      return StatsEvent.Total(mean: round2(meanValue), p99: round2(p99), max: round2(Double(sorted.last!) / 1000))
    }
    let total = mean(\.totalUs)
    let gpu = mean(\.gpuUs)
    let queue = mean(\.queueUs)
    return StatsEvent(
      frames: framesTotal,
      fps: round2(n / seconds),
      servedMs: spread(rows.map { $0.totalUs + $0.sendUs }),
      stagesMs: StatsEvent.Stages(queue: round2(queue), gpu: round2(gpu), other: round2(max(0, total - gpu - queue)), send: round2(mean(\.sendUs))),
      slow: rows.filter { $0.totalUs > FrameStats.slowUs }.count,
      windowS: pythonRound(seconds, 1))
  }
}

/// `round(value, 2)`, as the Python server rounds what it publishes.
func round2(_ value: Double) -> Double {
  pythonRound(value, 2)
}

/// Process-wide owner of the loaded engine and of the build in flight: the
/// Swift form of `session.EngineHost`.
///
/// The engine outlives the connection. The comma reconnects at every
/// handover, and reloading a 770 MB model costs seconds of modeld's budget.
/// Everything that touches `loaded` holds `lock`: the job thread swaps engines
/// while the request loop runs frames.
public final class EngineHost: @unchecked Sendable {
  static let progressInterval: TimeInterval = 0.25

  let cache: EngineCache
  let lock = NSLock()
  var loaded: Loaded?
  var job: Job?
  /// Who hears about progress and completion over the link.
  var session: Session?
  /// A benchmark owns the engine: frames from a comma are answered NOT_READY.
  var benchmarking = false
  let frameStats = FrameStats()
  let log = ServerLog(category: "server")

  private var lastProgress: TimeInterval = 0
  private var lastStage: (stage: String?, frac: Double, msg: String) = (nil, 0, "")
  private var lastEngine: EngineEvent?
  private let listenersLock = NSLock()
  private var listeners: [@Sendable (HostEvent) -> Void] = []
  private let emitLock = NSLock()

  public init(cache: EngineCache) {
    self.cache = cache
  }

  var backend: any EngineBackend { cache.backend }

  // MARK: listeners

  public func subscribe(_ listener: @escaping @Sendable (HostEvent) -> Void) {
    listenersLock.lock()
    listeners.append(listener)
    listenersLock.unlock()
  }

  /// An engine event identical to the last one is dropped: a job finishing
  /// emits from the progress call and again from its end, and the two are the
  /// same event.
  func emit(_ event: HostEvent) {
    emitLock.lock()
    if case .engine(let snapshot) = event {
      if snapshot == lastEngine {
        emitLock.unlock()
        return
      }
      lastEngine = snapshot
    }
    emitLock.unlock()
    listenersLock.lock()
    let current = listeners
    listenersLock.unlock()
    for listener in current {
      listener(event)
    }
  }

  /// The `engine` event: what is loaded, or what is being prepared.
  public func snapshot() -> EngineEvent {
    lock.lock()
    defer { lock.unlock() }
    return snapshotLocked()
  }

  private func snapshotLocked() -> EngineEvent {
    let (stage, frac, msg) = lastStage
    if let loaded {
      return EngineEvent(state: .ready, sha256: loaded.sha256, detail: "", stage: nil, frac: 1, msg: msg, loadOnly: job?.loadOnly ?? false)
    }
    if let job, job.state == .building {
      return EngineEvent(
        state: job.loadOnly ? .loading : .building, sha256: job.sha256, detail: job.detail, stage: stage, frac: frac, msg: msg, loadOnly: job.loadOnly)
    }
    if let job, job.state == .failed {
      return EngineEvent(state: .failed, sha256: job.sha256, detail: job.detail, stage: "failed", frac: frac, msg: msg, loadOnly: job.loadOnly)
    }
    return .none
  }

  // MARK: what a client sees

  /// The engine state for one model, in the shape ENGINE_RESP carries. frame_skip
  /// is part of the identity: the spec handed back is stamped with it.
  func status(_ sha256: String?, frameSkip: Int? = nil) -> [String: Any] {
    lock.lock()
    defer { lock.unlock() }
    return statusLocked(sha256, frameSkip: frameSkip)
  }

  private func statusLocked(_ sha256: String?, frameSkip: Int? = nil) -> [String: Any] {
    let chunk = ModelConstants.chunk
    guard let sha256 else {
      return ["state": "none", "detail": "", "sha256": NSNull(), "chunk": chunk]
    }
    if let loaded, loaded.sha256 == sha256, frameSkip == nil || loaded.spec.frameSkip == frameSkip {
      return ["state": "ready", "detail": "", "sha256": sha256, "chunk": chunk, "spec": loaded.spec.dictionary()]
    }
    if let job, job.sha256 == sha256, job.state != .ready {
      return ["state": job.state.rawValue, "detail": job.detail, "sha256": sha256, "chunk": chunk]
    }
    if let job, job.state == .building {
      return ["state": "building", "sha256": sha256, "chunk": chunk, "detail": "another build is in progress (\(job.sha256.prefix(16)))"]
    }
    if cachedSpec(cache.entry(sha256)) != nil {
      // Built already, just not loaded. modeld, which never carries the ONNX,
      // would read need_upload as an engine that is gone.
      return ["state": "building", "sha256": sha256, "chunk": chunk, "detail": "engine cached, not loaded yet"]
    }
    return ["state": "need_upload", "sha256": sha256, "chunk": chunk, "detail": "have \(modelBytes(sha256)) of the model"]
  }

  public func loadedSHA() -> String? {
    lock.lock()
    defer { lock.unlock() }
    return loaded?.sha256
  }

  // MARK: requests

  /// Make `request` the model being served, starting whatever that takes.
  /// A control-channel prepare passes no session, and must not detach the
  /// comma's session from the progress it is waiting on.
  func request(_ request: Request, session: Session?) -> [String: Any] {
    lock.lock()
    if let session {
      self.session = session
    }
    if let loaded, loaded.sha256 == request.sha256, loaded.spec.frameSkip == request.frameSkip {
      let ready = readyStatus(loaded)
      lock.unlock()
      return ready
    }
    if let job, job.state == .building {
      let status = statusLocked(request.sha256, frameSkip: request.frameSkip)
      lock.unlock()
      return status
    }
    lock.unlock()
    let entry = cache.entry(request.sha256)
    let modelPath = cache.modelPath(request.sha256)
    let spec = specOnDisk(entry, modelPath: modelPath, request: request)
    if entry.exists, let spec {
      start(Job(sha256: request.sha256, loadOnly: true), request: request, entry: entry, modelPath: modelPath, spec: spec)
    } else if modelComplete(modelPath, nbytes: request.nbytes) {
      start(Job(sha256: request.sha256, loadOnly: false), request: request, entry: entry, modelPath: modelPath, spec: spec)
    } else {
      lock.lock()
      if let job, job.sha256 == request.sha256, job.state == .failed {
        // Whatever failed has left the disk; what the client needs now is need_upload.
        self.job = nil
      }
      lock.unlock()
    }
    return status(request.sha256, frameSkip: request.frameSkip)
  }

  private func readyStatus(_ loaded: Loaded) -> [String: Any] {
    ["state": "ready", "detail": "", "sha256": loaded.sha256, "chunk": ModelConstants.chunk, "spec": loaded.spec.dictionary()]
  }

  /// The spec a cached artifact's sidecar carries, if it has one.
  func cachedSpec(_ entry: CacheEntry) -> [String: Any]? {
    guard entry.exists else { return nil }
    guard let meta = try? entry.meta() else {
      log.warning("unreadable sidecar for \(entry.path.lastPathComponent)")
      return nil
    }
    return meta["spec"] as? [String: Any]
  }

  private func modelBytes(_ sha256: String) -> Int64 {
    fileSize(cache.modelPath(sha256))
  }

  /// Start loading whatever was loaded last, before a client asks for it.
  public func preload() {
    guard let (sha256, frameSkip) = cache.lastLoaded() else { return }
    lock.lock()
    let busy = loaded != nil || job != nil
    lock.unlock()
    if busy { return }
    let entry = cache.entry(sha256)
    guard let d = cachedSpec(entry), let spec = try? ModelSpec.from(d).withFrameSkip(frameSkip),
      let request = try? Request(sha256: sha256, nbytes: 0, frameSkip: frameSkip)
    else { return }
    log.info("preloading the engine loaded last: \(entry.path.lastPathComponent)")
    start(Job(sha256: sha256, loadOnly: true), request: request, entry: entry, modelPath: cache.modelPath(sha256), spec: spec)
  }

  /// The spec for a cached artifact, from its sidecar or failing that the ONNX.
  /// Only a whole model file is parsed.
  private func specOnDisk(_ entry: CacheEntry, modelPath: URL, request: Request) -> ModelSpec? {
    if let d = cachedSpec(entry), let spec = try? ModelSpec.from(d) {
      return spec.withFrameSkip(request.frameSkip)
    }
    if modelComplete(modelPath, nbytes: request.nbytes) {
      do {
        return try backend.deriveSpec(model: modelPath, sha256: request.sha256, nbytes: request.nbytes, frameSkip: request.frameSkip)
      } catch {
        log.error("could not derive a spec from \(modelPath.lastPathComponent): \(String(describing: error))")
      }
    }
    return nil
  }

  // MARK: the worker

  private func start(_ job: Job, request: Request, entry: CacheEntry, modelPath: URL, spec: ModelSpec?) {
    job.detail = job.loadOnly ? "loading engine" : "building engine"
    lock.lock()
    self.job = job
    // The stage belongs to the job: a fresh job must not show the last one's 100 %.
    lastStage = (nil, 0, "")
    lock.unlock()
    let thread = Thread { [self] in run(job, request: request, entry: entry, modelPath: modelPath, spec: spec) }
    thread.name = "jetlink-build"
    thread.qualityOfService = .userInitiated
    thread.stackSize = 8 << 20
    thread.start()
    emit(.engine(snapshot()))
  }

  private func run(_ job: Job, request: Request, entry: CacheEntry, modelPath: URL, spec: ModelSpec?) {
    var engine: (any Engine)?
    do {
      // One engine resident at a time: a build needs the memory.
      unload()
      var spec = spec
      if !job.loadOnly {
        spec = try buildJob(request, entry: entry, modelPath: modelPath, spec: spec)
      }
      guard var ready = spec else { throw HostError.failed("no model spec") }
      writeSpec(entry, ready)
      progress("load", 0, "deserializing engine", force: true)
      do {
        engine = try backend.load(artifact: entry.path, report: progressFn)
      } catch let invalid as ArtifactInvalid {
        // Wrong on disk, not wrong here. Replace it from the ONNX when there is
        // one, else let the client upload again.
        log.warning("discarding \(entry.path.lastPathComponent): \(invalid.description)")
        entry.remove()
        let size = fileSize(modelPath)
        let have = size > 0 && (request.nbytes == 0 || size == request.nbytes)
        if job.loadOnly && !have {
          throw HostError.failed("artifact invalid and the model is not on disk: \(invalid.description)")
        }
        ready = try buildJob(request, entry: entry, modelPath: modelPath, spec: ready)
        writeSpec(entry, ready)
        progress("load", 0, "deserializing engine", force: true)
        engine = try backend.load(artifact: entry.path, report: progressFn)
      }
      let loaded = try warm(engine!, spec: ready)
      engine = nil
      lock.lock()
      self.loaded = loaded
      job.state = .ready
      job.detail = ""
      lock.unlock()
      cache.rememberLoaded(request.sha256, frameSkip: request.frameSkip)
      progress("load", 1, "ready", force: true)
      log.info("engine ready: \(entry.path.lastPathComponent)")
    } catch {
      log.error("engine preparation failed: \(String(describing: error))")
      engine?.close()
      lock.lock()
      job.state = .failed
      job.detail = "\(type(of: error)): \(error)"
      lock.unlock()
      progress("failed", 1, job.detail, force: true)
    }
    lock.lock()
    let session = self.session
    lock.unlock()
    if let session {
      servePending(job, session: session)
      session.engineUpdate()
    }
    emit(.engine(snapshot()))
  }

  /// Start what the client is still waiting for, now the device is free. A
  /// preload that guessed the wrong sha holds the device while the client waits.
  private func servePending(_ done: Job, session: Session) {
    guard let request = session.request, done.sha256 != request.sha256 else { return }
    _ = self.request(request, session: session)
  }

  private func buildJob(_ request: Request, entry: CacheEntry, modelPath: URL, spec: ModelSpec?) throws -> ModelSpec {
    var spec = spec
    if spec == nil {
      progress("parse", 0, "reading model metadata", force: true)
      spec = try backend.deriveSpec(model: modelPath, sha256: request.sha256, nbytes: request.nbytes, frameSkip: request.frameSkip)
    }
    try backend.build(model: modelPath, artifact: entry.path, report: progressFn, metaExtra: ["spec": spec!.dictionary()])
    cache.prune(protect: entry.path)
    cache.sweepTemp()
    return spec!
  }

  private func writeSpec(_ entry: CacheEntry, _ spec: ModelSpec) {
    var meta = (try? entry.meta()) ?? [:]
    if meta["spec"] == nil {
      meta["spec"] = spec.dictionary()
      try? entry.writeMeta(meta)
    }
  }

  private func warm(_ engine: any Engine, spec: ModelSpec) throws -> Loaded {
    try checkShapes(engine, spec: spec)
    let staging = try Staging.forModel(spec, engine: engine)
    // Warm on zeros, so the first real frame pays for nothing lazy.
    let warped = [UInt8](repeating: 0, count: spec.warpedBytes)
    let packed = [Float](repeating: 0, count: spec.packedCount)
    try warped.withUnsafeBytes { w in
      try packed.withUnsafeBytes { p in
        try staging.stage(warped: w.baseAddress!, packed: p.baseAddress!)
      }
    }
    let warmed = try engine.warm()
    log.info("\(warmed)")
    staging.reset()
    return Loaded(sha256: spec.sha256, spec: spec, engine: engine, staging: staging)
  }

  /// Release the engine on a client's say-so.
  public func unload() {
    lock.lock()
    let old = loaded
    loaded = nil
    lock.unlock()
    if let old {
      old.engine.close()
      log.info("engine \(old.sha256.prefix(16)) unloaded")
      emit(.engine(snapshot()))
    }
  }

  public func close() {
    unload()
  }

  // MARK: talking back

  private var progressFn: ProgressFn {
    { [weak self] stage, frac, msg in self?.progress(stage, frac, msg) }
  }

  func progress(_ stage: String, _ frac: Double, _ msg: String, force: Bool = false) {
    let now = ProcessInfo.processInfo.systemUptime
    lock.lock()
    if !force && frac < 1 && now - lastProgress < EngineHost.progressInterval {
      lock.unlock()
      return
    }
    lastProgress = now
    lastStage = (stage, frac, msg)
    let session = self.session
    lock.unlock()
    emit(.progress(stage: stage, frac: pythonRound(frac, 4), msg: msg))
    session?.progress(stage, frac, msg)
  }
}

/// The engine is what will execute and the spec came from the file, so
/// disagreement means the artifact on disk is not this model's.
func checkShapes(_ engine: any Engine, spec: ModelSpec) throws {
  for (name, io) in engine.inputs {
    guard let want = spec.input(name) else {
      throw HostError.failed("engine input \(name) is not in the model spec")
    }
    if io.count != want.reduce(1, *) {
      throw HostError.failed("input \(name): engine \(io.shape) vs spec \(want)")
    }
  }
  let missing = Set(spec.inputShapes.map(\.name)).subtracting(engine.inputs.keys)
  if !missing.isEmpty {
    throw HostError.failed("engine has no input(s) \(missing.sorted()) the model spec declares")
  }
  guard let out = engine.outputs[ModelConstants.drivingOutput], out.count == spec.outputCount else {
    throw HostError.failed("output: engine \(engine.outputs[ModelConstants.drivingOutput]?.shape ?? []) vs spec \(spec.outputCount)")
  }
  let missingState = Set(spec.statePairs.map(\.output)).subtracting(engine.outputs.keys)
  if !missingState.isEmpty {
    throw HostError.failed("engine has no output(s) \(missingState.sorted()) to feed the state back from")
  }
}

func fileSize(_ url: URL) -> Int64 {
  ((try? FileManager.default.attributesOfItem(atPath: url.path))?[.size] as? NSNumber)?.int64Value ?? 0
}

/// Uploads land in place chunk by chunk, so the model file is only the model
/// once it is the size the client declared.
func modelComplete(_ url: URL, nbytes: Int64) -> Bool {
  var isDirectory: ObjCBool = false
  return FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory) && !isDirectory.boolValue && fileSize(url) == nbytes
}

/// The file's SHA-256 and size, read a megabyte at a time.
func sha256File(_ url: URL) throws -> (String, Int64) {
  let handle = try FileHandle(forReadingFrom: url)
  defer { try? handle.close() }
  var hasher = SHA256()
  var total: Int64 = 0
  while let chunk = try handle.read(upToCount: 1 << 20), !chunk.isEmpty {
    hasher.update(data: chunk)
    total += Int64(chunk.count)
  }
  return (hasher.finalize().map { String(format: "%02x", $0) }.joined(), total)
}
