import Foundation
import JetlinkKit

/// One client's connection: the request loop, the Swift form of
/// `session.Session`. One message at a time; builds run on the host's job
/// thread so progress keeps flowing. The inference path neither allocates
/// nor logs on the way to the reply.
final class Session: @unchecked Sendable {
  private let transport: TCPTransport
  private let host: EngineHost
  private let telemetry: () -> [String: Any]
  private let log = ServerLog(category: "session")
  private var client = ""
  private var lastSeq: UInt32 = 0
  private(set) var request: Request?
  private(set) var frames = 0

  /// The reply's float32 outputs, reused every frame.
  private var outputBuffer: UnsafeMutablePointer<Float>
  private var outputCapacity: Int
  /// The packed scalars, copied out of the receive buffer to align them.
  private var packedBuffer: UnsafeMutableRawPointer
  private var packedCapacity: Int
  /// The reply's fixed head, packed in place every frame.
  private let responseHead: UnsafeMutableRawPointer
  /// The reply's parts: head, outputs, and the telemetry when asked for.
  private let parts: UnsafeMutablePointer<UnsafeRawBufferPointer>

  init(transport: TCPTransport, host: EngineHost, telemetry: @escaping () -> [String: Any]) {
    self.transport = transport
    self.host = host
    self.telemetry = telemetry
    outputCapacity = 18_452
    outputBuffer = .allocate(capacity: outputCapacity)
    packedCapacity = 1 << 16
    packedBuffer = .allocate(byteCount: packedCapacity, alignment: 16)
    responseHead = .allocate(byteCount: Wire.inferRespSize, alignment: 8)
    parts = .allocate(capacity: 3)
    parts.initialize(repeating: UnsafeRawBufferPointer(start: nil, count: 0), count: 3)
  }

  deinit {
    outputBuffer.deallocate()
    packedBuffer.deallocate()
    responseHead.deallocate()
    parts.deallocate()
  }

  var peer: String { transport.peer }

  // MARK: plumbing

  private func send(_ type: Wire.Msg, seq: UInt32, parts: [UnsafeRawBufferPointer] = [], flags: Wire.Flag = []) throws {
    try transport.send(type, seq: seq, parts: parts, flags: flags)
  }

  private func sendJSON(_ type: Wire.Msg, seq: UInt32, _ object: [String: Any]) throws {
    try transport.sendJSON(type, seq: seq, object)
  }

  private func error(_ seq: UInt32, _ error: String, _ detail: String = "") throws {
    log.error("\(error): \(detail)")
    try sendJSON(.error, seq: seq, ["error": error, "detail": detail])
  }

  /// Sent from the job thread. The comma may have given up and fallen back;
  /// the build continues either way.
  func progress(_ stage: String, _ frac: Double, _ msg: String) {
    try? sendJSON(.progress, seq: 0, ["stage": stage, "frac": (frac * 10_000).rounded() / 10_000, "msg": msg])
  }

  /// The worker finished. Tell the client that is here now, whoever it is.
  func engineUpdate() {
    try? respondEngine(0)
  }

  /// The connection is gone. The engine stays.
  func close() {
    host.lock.lock()
    if host.session === self {
      host.session = nil
    }
    host.lock.unlock()
    transport.close()
  }

  /// Wakes the request loop from another thread, which then returns.
  func interrupt() {
    transport.shutdown()
  }

  // MARK: the loop

  /// Serves until the link fails, and returns why.
  func serveForever() -> String {
    while true {
      let message: Message
      do {
        message = try transport.recv()
      } catch let error as LinkError {
        if case .timedOut = error { continue }
        log.info("link closed: \(error.description)")
        return error.description
      } catch {
        return String(describing: error)
      }
      do {
        try handle(message)
      } catch let error as LinkError {
        return error.description
      } catch {
        // A bad request must not take the server down.
        log.error("handler failed: \(String(describing: error))")
        if (try? self.error(message.seq, String(describing: type(of: error)), String(describing: error))) == nil {
          return "could not report an error to the client"
        }
      }
    }
  }

  func handle(_ message: Message) throws {
    guard let type = Wire.Msg(rawValue: message.msgType) else {
      try error(message.seq, "unknown_message", "type \(message.msgType)")
      return
    }
    if type == .helloReq {
      // A hello means "a new client process", answered whatever the seq says.
      greet(message)
      try onHello(message)
      return
    }
    // Seqs never repeat on a connection, so anything at or below the last one
    // is a replay; running it would push the same image into the queues twice.
    if message.seq <= lastSeq {
      log.warning("dropping replayed message type=\(message.msgType) seq=\(message.seq) (last \(lastSeq))")
      return
    }
    lastSeq = message.seq
    switch type {
    case .inferReq: try onInfer(message)
    case .ping: try send(.pong, seq: message.seq)
    case .engineReq: try onEngineReq(message)
    case .uploadChunk: try onUploadChunk(message)
    case .uploadDone: try onUploadDone(message)
    case .stateReq: try onState(message)
    case .shutdownReq: try onShutdown(message)
    default: try error(message.seq, "unknown_message", "type \(message.msgType)")
    }
  }

  private func wanted() -> (String?, Int?) {
    guard let request else { return (nil, nil) }
    return (request.sha256, request.frameSkip)
  }

  private func greet(_ message: Message) {
    var who = ""
    if let object = JSONLine.decode(message.payload), let d = object["client"] as? [String: Any] {
      let name = (d["name"] as? String).flatMap { $0.isEmpty ? nil : $0 } ?? "client"
      let nonce = d["nonce"].map { "\($0)" } ?? "?"
      who = "\(name)/\(nonce)"
    }
    if !client.isEmpty && who != client {
      log.info("session handed from \(client) to \(who.isEmpty ? "an unnamed client" : who)")
    }
    client = who
    lastSeq = message.seq
    request = nil
    frames = 0
    log.info("hello from \(who.isEmpty ? "an unnamed client" : who) (seq \(message.seq))")
  }

  private func onHello(_ message: Message) throws {
    let (sha, skip) = wanted()
    var response: [String: Any] = [
      "protocol": Int(Wire.version),
      "engine_state": host.status(sha, frameSkip: skip)["state"] ?? "none",
      "loaded": host.loadedSHA() ?? NSNull(),
      "frames_served": frames,
      "cached_models": host.cache.inventory(),
      "telemetry": telemetry(),
      // This server never suspends: the comma holds the link while parked.
      "sleep_after": 0.0,
    ]
    for (key, value) in host.backend.describe() {
      response[key] = value
    }
    try sendJSON(.helloResp, seq: message.seq, response)
  }

  private func onEngineReq(_ message: Message) throws {
    guard let d = JSONLine.decode(message.payload), let sha = d["sha256"] as? String,
      let nbytes = (d["nbytes"] as? NSNumber)?.int64Value
    else {
      try error(message.seq, "bad_request", "ENGINE_REQ needs sha256 and nbytes")
      return
    }
    let skip = (d["frame_skip"] as? NSNumber)?.intValue ?? ModelConstants.defaultFrameSkip
    let request = try Request(sha256: sha, nbytes: nbytes, frameSkip: skip)
    self.request = request
    try sendJSON(.engineResp, seq: message.seq, host.request(request, session: self))
  }

  private func respondEngine(_ seq: UInt32) throws {
    let (sha, skip) = wanted()
    try sendJSON(.engineResp, seq: seq, host.status(sha, frameSkip: skip))
  }

  private func onUploadChunk(_ message: Message) throws {
    guard let request else {
      try error(message.seq, "no_model", "send ENGINE_REQ first")
      return
    }
    guard message.payload.count >= 8 else {
      try error(message.seq, "bad_upload", "missing chunk offset")
      return
    }
    let offset = UInt64(littleEndian: message.payload.loadUnaligned(as: UInt64.self))
    let data = UnsafeRawBufferPointer(rebasing: message.payload[8...])
    if offset + UInt64(data.count) > UInt64(request.nbytes) {
      try error(message.seq, "bad_upload", "chunk exceeds declared model size")
      return
    }
    let path = host.cache.modelPath(request.sha256)
    if offset == 0 || !FileManager.default.fileExists(atPath: path.path) {
      FileManager.default.createFile(atPath: path.path, contents: nil)
    }
    let handle = try FileHandle(forWritingTo: path)
    defer { try? handle.close() }
    if offset == 0 {
      try handle.truncate(atOffset: 0)
    }
    try handle.seek(toOffset: offset)
    try handle.write(contentsOf: data)
    // Silent on purpose: the client streams chunks without reading between them.
  }

  private func onUploadDone(_ message: Message) throws {
    guard let request else {
      try error(message.seq, "no_model", "send ENGINE_REQ first")
      return
    }
    let path = host.cache.modelPath(request.sha256)
    let digest = (try? sha256File(path))?.0
    if digest != request.sha256 {
      try? FileManager.default.removeItem(at: path)
      try sendJSON(
        .engineResp, seq: message.seq,
        [
          "state": "failed", "detail": "sha256 mismatch after upload", "sha256": request.sha256, "chunk": ModelConstants.chunk,
        ])
      return
    }
    progress("upload", 1, "verified")
    try sendJSON(.engineResp, seq: message.seq, host.request(request, session: self))
  }

  // MARK: the hot path

  private func onInfer(_ message: Message) throws {
    let (sha, skip) = wanted()
    host.lock.lock()
    defer { host.lock.unlock() }
    guard let loaded = host.loaded, loaded.sha256 == sha, loaded.spec.frameSkip == skip, !host.benchmarking else {
      try respond(message.seq, frameID: 0, status: .notReady, gpuUs: 0, queueUs: 0, totalUs: 0, outputBytes: 0, state: nil)
      return
    }
    try infer(loaded, message)
  }

  /// INFER_RESP: the head, `outputBytes` of the output buffer, and the
  /// telemetry, in one write from buffers this session owns.
  private func respond(
    _ seq: UInt32, frameID: UInt32, status: Wire.Status, gpuUs: UInt32, queueUs: UInt32, totalUs: UInt32, outputBytes: Int, state: Data?
  ) throws {
    Wire.packInferResp(frameID: frameID, status: status, gpuUs: gpuUs, queueUs: queueUs, totalUs: totalUs, into: responseHead)
    parts[0] = UnsafeRawBufferPointer(start: responseHead, count: Wire.inferRespSize)
    parts[1] = UnsafeRawBufferPointer(start: outputBuffer, count: outputBytes)
    if let state {
      try state.withUnsafeBytes { bytes in
        parts[2] = bytes
        try transport.send(.inferResp, seq: seq, parts: UnsafeBufferPointer(start: parts, count: 3))
      }
    } else {
      try transport.send(.inferResp, seq: seq, parts: UnsafeBufferPointer(start: parts, count: 2))
    }
  }

  private func infer(_ loaded: Loaded, _ message: Message) throws {
    let started = DispatchTime.now().uptimeNanoseconds
    let spec = loaded.spec
    guard message.payload.count == spec.inferReqBytes else {
      // The offsets below come from the spec, not the wire: a client on
      // another model would have its scalars read out of the image.
      try respond(message.seq, frameID: 0, status: .badShape, gpuUs: 0, queueUs: 0, totalUs: 0, outputBytes: 0, state: nil)
      return
    }
    let base = message.payload.baseAddress!
    let frameID = UInt32(littleEndian: base.loadUnaligned(as: UInt32.self))
    let flags = Wire.Flag(rawValue: UInt32(littleEndian: base.loadUnaligned(fromByteOffset: 4, as: UInt32.self)))
    if flags.contains(.resetQueues) {
      loaded.staging.reset()
    }
    let warped = base + Wire.inferReqSize
    if spec.packedBytes > packedCapacity {
      packedBuffer.deallocate()
      packedCapacity = spec.packedBytes
      packedBuffer = .allocate(byteCount: packedCapacity, alignment: 16)
    }
    packedBuffer.copyMemory(from: warped + spec.warpedBytes, byteCount: spec.packedBytes)

    var status = Wire.Status.ok
    var queueUs: UInt32 = 0
    do {
      try loaded.staging.stage(warped: warped, packed: packedBuffer)
      queueUs = microseconds(since: started)
      try loaded.engine.run()
    } catch {
      log.error("inference failed: \(String(describing: error))")
      status = .inferFailed
    }

    let count = spec.outputCount
    if count > outputCapacity {
      outputBuffer.deallocate()
      outputCapacity = count
      outputBuffer = .allocate(capacity: count)
    }
    if status == .ok, let io = loaded.engine.outputs[ModelConstants.drivingOutput], let out = loaded.engine.output(ModelConstants.drivingOutput) {
      // float32 on the wire whatever the graph says; openpilot drops to the
      // small model on a non-finite output either way, so say so here.
      var finite = true
      switch io.type {
      case .float:
        outputBuffer.update(from: out.assumingMemoryBound(to: Float.self), count: count)
      case .float16:
        Convert.f16ToF32(out, outputBuffer, count: count)
      default:
        status = .inferFailed
      }
      for i in 0..<count where !outputBuffer[i].isFinite {
        finite = false
        break
      }
      if !finite && status == .ok {
        status = .notFinite
      }
    } else if status == .ok {
      status = .inferFailed
    }

    let totalUs = microseconds(since: started)
    let state: Data? = flags.contains(.wantState) ? JSONLine.encode(telemetry()) : nil
    let sendStarted = DispatchTime.now().uptimeNanoseconds
    try respond(
      message.seq, frameID: frameID, status: status, gpuUs: loaded.engine.lastGpuUs, queueUs: queueUs, totalUs: totalUs,
      outputBytes: status == .ok || status == .notFinite ? count * 4 : 0, state: state)
    let sendUs = microseconds(since: sendStarted)
    frames += 1
    if totalUs > FrameStats.slowUs || sendUs > 10_000 {
      log.warning(
        "slow frame \(frameID): gpu \(Double(loaded.engine.lastGpuUs) / 1000) queue \(Double(queueUs) / 1000) total \(Double(totalUs) / 1000) send \(Double(sendUs) / 1000) ms"
      )
    }
    host.frameStats.record(totalUs: totalUs, gpuUs: loaded.engine.lastGpuUs, queueUs: queueUs, sendUs: sendUs)
  }

  private func onShutdown(_ message: Message) throws {
    // A phone does not power itself off for the comma; say so rather than pretend.
    try sendJSON(.shutdownResp, seq: message.seq, ["ok": false, "detail": "this server cannot power its device off"])
  }

  private func onState(_ message: Message) throws {
    let (sha, skip) = wanted()
    let status = host.status(sha, frameSkip: skip)
    var response = telemetry()
    response["engine_state"] = status["state"] ?? "none"
    response["detail"] = status["detail"] ?? ""
    response["loaded"] = host.loadedSHA() ?? NSNull()
    response["frames_served"] = frames
    try sendJSON(.stateResp, seq: message.seq, response)
  }
}

@inline(__always)
func microseconds(since start: UInt64) -> UInt32 {
  UInt32(min(UInt64(UInt32.max), (DispatchTime.now().uptimeNanoseconds - start) / 1000))
}
