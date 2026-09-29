import Foundation
import JetlinkKit
import Testing

@testable import JetlinkServer

/// The host's share of a frame, timed around an engine whose run does
/// nothing: `stage` is the queues alone, `frame` the session's whole INFER
/// path (parse, stage, run, float32 outputs, finite check, reply) to a link
/// that drops the reply. The shapes are two 766 MB models', every tensor
/// float16 as a TensorRT plan takes them: BMRLNAP v6 (queued) and Cinque
/// Terre V3 (stateful). Off unless JETLINK_BENCH is set, and only meaningful
/// optimized:
///
///     JETLINK_BENCH=1 swift test -c release -Xswiftc -enable-testing --filter StagingBenchmark
@Suite("Staging benchmark", .serialized, .enabled(if: ProcessInfo.processInfo.environment["JETLINK_BENCH"] != nil))
struct StagingBenchmark {
  static let frames = 3000
  static let warmup = 300

  @Test("A queued 766 MB model")
  func queued() throws {
    try StagingBenchmark.run(
      "queued",
      inputs: [
        NamedShape("img", [1, 12, 128, 256]), NamedShape("big_img", [1, 12, 128, 256]), NamedShape("desire_pulse", [1, 33, 8]),
        NamedShape("traffic_convention", [1, 2]), NamedShape("action_t", [1, 2]), NamedShape("features_buffer", [1, 32, 32, 512]),
      ],
      outputs: [NamedShape("outputs", [1, 18452])])
  }

  @Test("A stateful 766 MB model")
  func stateful() throws {
    try StagingBenchmark.run(
      "stateful",
      inputs: [
        NamedShape("new_img", [2, 6, 128, 256]), NamedShape("desire", [8]), NamedShape("traffic_convention", [1, 2]),
        NamedShape("action_t", [1, 2]), NamedShape("state_img_q", [2, 5, 6, 128, 256]), NamedShape("state_desire_q", [132, 1, 8]),
        NamedShape("state_feat_q", [128, 1, 16384]),
      ],
      outputs: [
        NamedShape("outputs", [1, 18452]), NamedShape("next_state_img_q", [2, 5, 6, 128, 256]),
        NamedShape("next_state_desire_q", [132, 1, 8]), NamedShape("next_state_feat_q", [128, 1, 16384]),
      ])
  }

  static func run(_ label: String, inputs: [NamedShape], outputs: [NamedShape]) throws {
    let sha = String(repeating: "ab", count: 32)
    let spec = ModelSpec(
      sha256: sha, nbytes: 765_955_335, frameSkip: 4, inputShapes: inputs, outputShapes: outputs,
      outputSlices: [NamedRange("hidden_state", 2066..<18450)], checkpoint: nil)
    let tensors = { (shapes: [NamedShape]) in
      Dictionary(uniqueKeysWithValues: shapes.map { ($0.name, TensorSpec(name: $0.name, type: .float16, shape: $0.shape)) })
    }
    let engine = try IdleEngine(inputs: tensors(inputs), outputs: tensors(outputs))
    let staging = try Staging.forModel(spec, engine: engine)
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("staging-bench-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: root) }
    let host = EngineHost(cache: try ServerCache(root: root, backend: cpuBackend()))
    host.loaded = Loaded(sha256: sha, spec: spec, engine: engine, staging: staging)
    let session = Session(transport: DiscardingLink(), host: host)
    let request = JSONLine.encode(["sha256": sha, "nbytes": spec.nbytes, "frame_skip": spec.frameSkip])
    try request.withUnsafeBytes { try session.handle(Message(msgType: Wire.Msg.engineReq.rawValue, seq: 1, flags: 0, payload: $0)) }

    // An INFER_REQ where the USB and TCP readers leave one: at the start of
    // a 64-byte aligned buffer, after the header.
    let payloadBytes = spec.inferReqBytes
    let rx = UnsafeMutableRawPointer.allocate(byteCount: Wire.headerSize + payloadBytes, alignment: 64)
    defer { rx.deallocate() }
    let payload = rx + Wire.headerSize
    var generator = SystemRandomNumberGenerator()
    for i in 0..<spec.warpedBytes {
      payload.storeBytes(of: UInt8.random(in: 0...255, using: &generator), toByteOffset: Wire.inferReqSize + i, as: UInt8.self)
    }
    let warped = payload + Wire.inferReqSize
    let packed = warped + spec.warpedBytes
    for i in 0..<spec.packedCount {
      packed.storeBytes(of: Float.random(in: -4...4, using: &generator), toByteOffset: i * 4, as: Float.self)
    }

    var stage: [Double] = []
    var frame: [Double] = []
    var seq: UInt32 = 1
    for i in 0..<(warmup + frames) {
      var started = DispatchTime.now().uptimeNanoseconds
      try staging.stage(warped: warped, packed: packed)
      let stageNs = DispatchTime.now().uptimeNanoseconds - started
      seq += 1
      payload.storeBytes(of: UInt32(i).littleEndian, as: UInt32.self)
      payload.storeBytes(of: UInt32(0), toByteOffset: 4, as: UInt32.self)
      started = DispatchTime.now().uptimeNanoseconds
      try session.handle(Message(msgType: Wire.Msg.inferReq.rawValue, seq: seq, flags: 0, payload: UnsafeRawBufferPointer(start: payload, count: payloadBytes)))
      let frameNs = DispatchTime.now().uptimeNanoseconds - started
      if i >= warmup {
        stage.append(Double(stageNs) / 1000)
        frame.append(Double(frameNs) / 1000)
      }
    }
    #expect(session.frames == warmup + frames)
    print("staging benchmark \(label): stage \(summary(stage)); frame \(summary(frame)) (us over \(frames) frames)")
  }

  static func summary(_ values: [Double]) -> String {
    let sorted = values.sorted()
    let mean = sorted.reduce(0, +) / Double(sorted.count)
    let p50 = sorted[sorted.count / 2]
    let p99 = sorted[Int(0.99 * Double(sorted.count - 1))]
    return String(format: "mean %.1f p50 %.1f p99 %.1f max %.1f", mean, p50, p99, sorted.last!)
  }
}

/// Host buffers and nothing to run, looping any state it is given.
final class IdleEngine: EngineCore, @unchecked Sendable {
  override func bindLoop(_ pairs: [(input: String, output: String)]) throws -> Bool { true }
  override func execute() throws {}
}

/// A link whose replies go nowhere.
final class DiscardingLink: MessageLink {
  var peer: String { "bench" }
  var medium: LinkMedium? { nil }
  var connectsOnOpen: Bool { false }
  func recv() throws -> Message { throw LinkError.closed("bench") }
  func sendParts(_ type: Wire.Msg, seq: UInt32, parts: UnsafeBufferPointer<UnsafeRawBufferPointer>, flags: Wire.Flag) throws {}
  func shutdown() {}
  func close() {}
}
