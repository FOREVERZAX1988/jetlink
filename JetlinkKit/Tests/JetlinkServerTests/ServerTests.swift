import Foundation
import Testing
@testable import JetlinkServer

/// The whole server, over a real socket, on onnxruntime's CPU provider: the
/// upload, the Swift preparation, the build, the load, the queues or the state
/// loop, and the reply, checked bit for bit against what the Python server's
/// parts compute for the same frames.
@Suite("Server", .serialized)
struct ServerTests {
  func serve(_ body: (Server, TestClient) throws -> Void) throws {
    let cache = try TemporaryDirectory()
    let server = try Server(
      configuration: Server.Configuration(host: "127.0.0.1", port: 0, cacheRoot: cache.url, device: .cpu, keepAlive: false, preload: false),
      preparer: ONNXPreparer())
    try server.start()
    defer { server.stop() }
    let client = try TestClient(port: server.port!)
    defer { client.close() }
    try body(server, client)
  }

  @Test("A comma is served the outputs the Python server computes", arguments: ["tiny_queued", "tiny_stateful"])
  func servesGoldenFrames(_ name: String) throws {
    let golden = try Golden(name)
    try serve { server, client in
      try client.send(.helloReq, JSONSerialization.data(withJSONObject: ["client": ["name": "test", "nonce": 1]]))
      let hello = try client.recv(.helloResp).json
      #expect(hello["protocol"] as? Int == 2)
      #expect(hello["backend"] as? String == "ort")

      let ready = try client.ensureEngine(model: golden.model, sha256: golden.sha256)
      let spec = try ModelSpec.from(ready["spec"] as! [String: Any])
      let frameBytes = spec.warpedBytes + spec.packedBytes
      let count = golden.frames.count / frameBytes
      #expect(count == 8)

      for i in 0..<count {
        var request = withUnsafeBytes(of: UInt32(i).littleEndian) { Data($0) }
        request.append(contentsOf: withUnsafeBytes(of: UInt32(0).littleEndian) { Data($0) })
        request.append(golden.frames[(i * frameBytes)..<((i + 1) * frameBytes)])
        try client.send(.inferReq, request)
        let reply = try client.recv(.inferResp)
        let status = reply.payload.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 4, as: UInt32.self) }
        #expect(status == Wire.Status.ok.rawValue)
        let outputs = reply.payload[Wire.inferRespSize...]
        let expected = golden.expected[(i * spec.outputBytes)..<((i + 1) * spec.outputBytes)]
        #expect(Data(outputs) == Data(expected), "frame \(i) differs from Python's")
      }
      #expect(server.framesServed == count)
    }
  }

  @Test("A replayed request is dropped, and pings are answered")
  func dropsReplays() throws {
    try serve { _, client in
      let seq = try client.send(.ping)
      #expect(try client.recv().type == Wire.Msg.pong.rawValue)
      try client.send(.ping, seq: seq)   // a replay: no answer
      let next = try client.send(.ping)
      let reply = try client.recv()
      #expect(reply.type == Wire.Msg.pong.rawValue)
      #expect(reply.seq == next)
    }
  }

  @Test("A frame with no engine is answered NOT_READY, not dropped")
  func notReady() throws {
    try serve { _, client in
      try client.send(.inferReq, Data(count: 64))
      let reply = try client.recv(.inferResp)
      let status = reply.payload.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 4, as: UInt32.self) }
      #expect(status == Wire.Status.notReady.rawValue)
    }
  }

  @Test("A model that does not hash to its name is refused")
  func refusesACorruptUpload() throws {
    let golden = try Golden("tiny_queued")
    try serve { server, client in
      let bytes = try Data(contentsOf: golden.model)
      try client.sendJSON(.engineReq, ["sha256": golden.sha256, "nbytes": bytes.count, "frame_skip": 4])
      #expect(try client.recv(.engineResp).json["state"] as? String == "need_upload")
      var payload = withUnsafeBytes(of: UInt64(0).littleEndian) { Data($0) }
      payload.append(Data(repeating: 7, count: bytes.count))
      try client.send(.uploadChunk, payload)
      try client.sendJSON(.uploadDone, ["sha256": golden.sha256])
      let reply = try client.recv(.engineResp).json
      #expect(reply["state"] as? String == "failed")
      #expect(!FileManager.default.fileExists(atPath: server.cache.modelPath(golden.sha256).path))
    }
  }

  @Test("A new connection takes over from the one being served")
  func newConnectionTakesOver() throws {
    try serve { server, first in
      try first.send(.ping)
      #expect(try first.recv().type == Wire.Msg.pong.rawValue)
      let second = try TestClient(port: server.port!)
      defer { second.close() }
      try second.send(.ping)
      #expect(try second.recv().type == Wire.Msg.pong.rawValue)
      #expect(throws: LinkError.self) { try first.recv() }
    }
  }

  @Test("The engine outlives the connection, and the next comma gets it loaded")
  func engineOutlivesConnection() throws {
    let golden = try Golden("tiny_stateful")
    try serve { server, client in
      _ = try client.ensureEngine(model: golden.model, sha256: golden.sha256)
      client.close()
      let again = try TestClient(port: server.port!)
      defer { again.close() }
      let bytes = try Data(contentsOf: golden.model)
      try again.sendJSON(.engineReq, ["sha256": golden.sha256, "nbytes": bytes.count, "frame_skip": 4])
      #expect(try again.recv(.engineResp).json["state"] as? String == "ready")
    }
  }
}
