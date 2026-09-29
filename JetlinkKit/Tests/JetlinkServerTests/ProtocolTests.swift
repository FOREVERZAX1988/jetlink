import Foundation
import JetlinkKit
import Testing

@testable import JetlinkServer

/// Protocol 3 on the server's side: the hidden state stays here, and a comma
/// of another version is told which side to update, in one round trip and
/// with the stream still in sync. tests/test_swift_server.py plays the same
/// comma from its bytes over TCP against the built binary.
@Suite("Protocol 3", .serialized)
struct ProtocolTests {
  /// A session over a link that records what it sends.
  func session() throws -> (Session, RecordingLink, TemporaryDirectory) {
    let cache = try TemporaryDirectory()
    let host = EngineHost(cache: try ServerCache(root: cache.url, backend: cpuBackend()))
    let link = RecordingLink()
    return (Session(transport: link, host: host), link, cache)
  }

  func handle(_ session: Session, _ type: Wire.Msg, seq: UInt32, json: [String: Any] = [:], version: UInt16) throws {
    let payload = try JSONSerialization.data(withJSONObject: json)
    try payload.withUnsafeBytes { try session.handle(Message(msgType: type.rawValue, seq: seq, flags: 0, payload: $0, version: version)) }
  }

  @Test("A comma of protocol 2 is told to update its package, and its next try is answered the same")
  func oldComma() throws {
    let (session, link, _) = try session()
    // what the installed client's hello says: no protocol
    for seq: UInt32 in [1, 2] {
      try handle(session, .helloReq, seq: seq, json: ["client": ["name": "modeld", "nonce": "ab12"]], version: 2)
      let sent = try #require(link.last)
      #expect(sent.type == .error && sent.seq == seq)
      #expect(sent.json["error"] as? String == "protocol")
      #expect(sent.json["detail"] as? String == "this server runs jetlink protocol 3 and the comma 2: update the comma's jetlink package")
    }
    #expect(!link.sent.contains { $0.type == .helloResp })
  }

  @Test("A protocol 2 message outside the envelope is refused by name, and the session reads on")
  func oldFrame() throws {
    let (session, link, _) = try session()
    try handle(session, .ping, seq: 1, version: 2)
    #expect(link.last?.type == .error)
    #expect((link.last?.json["detail"] as? String)?.hasSuffix("update the comma's jetlink package") == true)
    try handle(session, .ping, seq: 2, version: 3)
    #expect(link.last?.type == .pong && link.last?.seq == 2)
  }

  @Test("A newer comma is told to update the server")
  func newerComma() throws {
    let (session, link, _) = try session()
    try handle(session, .helloReq, seq: 1, json: ["client": ["name": "modeld", "nonce": "1", "protocol": 4]], version: 2)
    #expect(link.last?.type == .error)
    #expect((link.last?.json["detail"] as? String)?.hasSuffix("update jetlink on this server") == true)
    try handle(session, .helloReq, seq: 2, json: ["client": ["name": "modeld", "nonce": "1", "protocol": 3]], version: 2)
    #expect(link.last?.type == .helloResp)
    #expect(link.last?.json["protocol"] as? Int == 3)
  }

  @Test("The low-battery shutdown is answered whatever the comma's version")
  func shutdownInTheEnvelope() throws {
    let (session, link, _) = try session()
    try handle(session, .shutdownReq, seq: 1, json: ["reason": "car battery"], version: 2)
    #expect(link.last?.type == .shutdownResp)
    #expect(link.last?.json["ok"] is Bool)
  }

  /// The queued golden frames carry a prev_feat of their own, as protocol 2
  /// sent it. Taken in through the server's own feedback path, as a hidden
  /// state kept from the frame before, they give protocol 2's outputs: the
  /// staging and the engine are unchanged, only where the hidden state comes
  /// from is.
  @Test("Protocol 2's outputs, from the server's own feedback path")
  func protocol2Outputs() throws {
    let golden = try Golden("tiny_queued")
    let cache = try TemporaryDirectory()
    let server = try Server(
      configuration: Server.Configuration(host: "127.0.0.1", port: 0, cacheRoot: cache.url, preload: false), backend: cpuBackend())
    try server.start()
    defer { server.stop() }
    let client = try TestClient(port: server.port!)
    let ready = try client.ensureEngine(model: golden.model, sha256: golden.sha256)
    client.close()
    let spec = try ModelSpec.from(ready["spec"] as! [String: Any])
    let hidden = try #require(spec.hiddenRange)

    server.host.lock.lock()
    defer { server.host.lock.unlock() }
    let loaded = try #require(server.host.loaded)
    let type = try #require(loaded.engine.outputs[ModelConstants.drivingOutput]?.type)
    var kept = [Float](repeating: 0, count: spec.outputCount)
    var output = [Float](repeating: 0, count: spec.outputCount)
    let frameBytes = golden.frameBytes(spec)
    loaded.staging.reset()
    for i in 0..<(golden.frames.count / frameBytes) {
      try golden.frames.withUnsafeBytes { raw in
        let frame = raw.baseAddress! + i * frameBytes
        let packed = frame + spec.warpedBytes
        // the recorded prev_feat, after the scalars, as the last frame's hidden state
        kept.withUnsafeMutableBytes { k in
          (k.baseAddress! + hidden.lowerBound * 4).copyMemory(from: packed + spec.packedBytes, byteCount: hidden.count * 4)
        }
        kept.withUnsafeBufferPointer { loaded.staging.keep(outputs: $0.baseAddress!) }
        try loaded.staging.stage(warped: frame, packed: packed)
      }
      try loaded.engine.run()
      let out = try #require(loaded.engine.output(ModelConstants.drivingOutput))
      output.withUnsafeMutableBytes { o in
        if type == .float16 {
          Convert.f16ToF32(out, o.baseAddress!, count: spec.outputCount)
        } else {
          o.baseAddress!.copyMemory(from: out, byteCount: spec.outputBytes)
        }
      }
      let got = output.withUnsafeBytes { Data($0) }
      let expected = Data(golden.expected[(i * spec.outputBytes)..<((i + 1) * spec.outputBytes)])
      #if os(Android) || os(Linux)
        // fp16 arithmetic there; see CommaClient.replay
        #expect(Golden.correlation(got, expected) >= 0.999, "frame \(i)")
      #else
        #expect(got == expected, "frame \(i) differs from protocol 2's by up to \(Golden.worstDifference(got, expected))")
      #endif
    }
  }
}

/// A link whose sends are kept, one message each.
final class RecordingLink: MessageLink, @unchecked Sendable {
  struct Sent {
    let type: Wire.Msg
    let seq: UInt32
    let payload: Data

    var json: [String: Any] { (try? JSONSerialization.jsonObject(with: payload)) as? [String: Any] ?? [:] }
  }

  private let lock = NSLock()
  private var messages: [Sent] = []

  var sent: [Sent] { lock.withLock { messages } }
  var last: Sent? { sent.last }

  var peer: String { "recording" }
  var medium: LinkMedium? { nil }
  var connectsOnOpen: Bool { true }
  func recv() throws -> Message { throw LinkError.closed("recording") }
  func sendParts(_ type: Wire.Msg, seq: UInt32, parts: UnsafeBufferPointer<UnsafeRawBufferPointer>, flags: Wire.Flag) throws {
    var payload = Data()
    for part in parts where part.count > 0 {
      payload.append(contentsOf: part)
    }
    lock.withLock { messages.append(Sent(type: type, seq: seq, payload: payload)) }
  }
  func shutdown() {}
  func close() {}
}
