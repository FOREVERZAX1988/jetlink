import Foundation
import Testing

@testable import JetlinkServer

/// The golden files make_server_fixtures.py writes, read in place.
enum Fixture {
  static let directory = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appending(path: "Fixtures", directoryHint: .isDirectory)

  static func url(_ name: String) -> URL {
    directory.appending(path: name)
  }

  static func data(_ name: String) throws -> Data {
    try Data(contentsOf: url(name))
  }

  static func json(_ name: String) throws -> [String: Any] {
    try JSONSerialization.jsonObject(with: data(name)) as! [String: Any]
  }
}

/// A fresh directory, removed when the test is done with it.
final class TemporaryDirectory {
  let url: URL

  init() throws {
    url = FileManager.default.temporaryDirectory.appending(path: "jetlink-tests-\(UUID().uuidString)", directoryHint: .isDirectory)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
  }

  deinit {
    try? FileManager.default.removeItem(at: url)
  }
}

/// A message the server sent.
struct Reply {
  let type: UInt16
  let seq: UInt32
  let payload: Data

  var json: [String: Any] {
    (try? JSONSerialization.jsonObject(with: payload)) as? [String: Any] ?? [:]
  }

  var status: UInt32 {
    payload.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 4, as: UInt32.self) }
  }
}

/// The comma's side of the wire, as much of it as the tests need: jetlink's
/// client.py, message by message, over TCP (`TestClient`) or the fake USB
/// pipes (`GadgetClient`).
protocol CommaClient: AnyObject {
  /// Sends one message with the next seq, or `explicit`; returns the seq.
  func sendMessage(_ type: Wire.Msg, _ payload: Data, flags: Wire.Flag, seq explicit: UInt32?) throws -> UInt32
  /// The next message the server sent, whatever it is.
  func recv() throws -> Reply
}

extension CommaClient {
  @discardableResult
  func send(_ type: Wire.Msg, _ payload: Data = Data(), flags: Wire.Flag = [], seq explicit: UInt32? = nil) throws -> UInt32 {
    try sendMessage(type, payload, flags: flags, seq: explicit)
  }

  @discardableResult
  func sendJSON(_ type: Wire.Msg, _ object: [String: Any]) throws -> UInt32 {
    try send(type, try JSONSerialization.data(withJSONObject: object))
  }

  /// The next reply of `type`, skipping progress on the way.
  func recv(_ type: Wire.Msg) throws -> Reply {
    while true {
      let reply = try recv()
      if reply.type == type.rawValue { return reply }
      if reply.type == Wire.Msg.progress.rawValue { continue }
      throw TestError("expected \(type), got message type \(reply.type): \(String(decoding: reply.payload, as: UTF8.self))")
    }
  }

  /// ENGINE_REQ, uploading the model when asked, until the engine is ready.
  func ensureEngine(model: URL, sha256: String) throws -> [String: Any] {
    let bytes = try Data(contentsOf: model)
    try sendJSON(.engineReq, ["sha256": sha256, "nbytes": bytes.count, "frame_skip": 4])
    var state = try recv(.engineResp).json
    if state["state"] as? String == "need_upload" {
      // Two chunks, so the offsets are exercised.
      let half = bytes.count / 2
      for (offset, chunk) in [(0, bytes[..<half]), (half, bytes[half...])] {
        var payload = withUnsafeBytes(of: UInt64(offset).littleEndian) { Data($0) }
        payload.append(contentsOf: chunk)
        try send(.uploadChunk, payload)
      }
      try sendJSON(.uploadDone, ["sha256": sha256])
      state = try recv(.engineResp).json
    }
    while state["state"] as? String == "building" {
      state = try recv(.engineResp).json
    }
    guard state["state"] as? String == "ready" else {
      throw TestError("engine not ready: \(state)")
    }
    return state
  }

  /// Hello, the model, then Python's golden frames, each reply checked bit
  /// for bit against Python's output. Returns the hello and the frames sent.
  @discardableResult
  func replay(_ golden: Golden) throws -> (hello: [String: Any], frames: Int) {
    try send(.helloReq, JSONSerialization.data(withJSONObject: ["client": ["name": "test", "nonce": 1]]))
    let hello = try recv(.helloResp).json
    let ready = try ensureEngine(model: golden.model, sha256: golden.sha256)
    let spec = try ModelSpec.from(ready["spec"] as! [String: Any])
    let frameBytes = spec.warpedBytes + spec.packedBytes
    let count = golden.frames.count / frameBytes
    for i in 0..<count {
      var request = withUnsafeBytes(of: UInt32(i).littleEndian) { Data($0) }
      request.append(contentsOf: withUnsafeBytes(of: UInt32(0).littleEndian) { Data($0) })
      request.append(golden.frames[(i * frameBytes)..<((i + 1) * frameBytes)])
      try send(.inferReq, request)
      let reply = try recv(.inferResp)
      #expect(reply.status == Wire.Status.ok.rawValue)
      let expected = golden.expected[(i * spec.outputBytes)..<((i + 1) * spec.outputBytes)]
      #expect(Data(reply.payload[Wire.inferRespSize...]) == Data(expected), "frame \(i) differs from Python's")
    }
    return (hello, count)
  }
}

/// The comma over TCP.
final class TestClient: CommaClient {
  let transport: TCPTransport
  private var seq: UInt32 = 0

  init(port: UInt16) throws {
    transport = try TCPTransport.connect(host: "127.0.0.1", port: port)
    transport.setReceiveTimeout(60)
  }

  func sendMessage(_ type: Wire.Msg, _ payload: Data, flags: Wire.Flag, seq explicit: UInt32?) throws -> UInt32 {
    let seq =
      explicit
      ?? {
        self.seq += 1
        return self.seq
      }()
    try transport.send(type, seq: seq, data: [payload], flags: flags)
    return seq
  }

  func recv() throws -> Reply {
    let message = try transport.recv()
    return Reply(type: message.msgType, seq: message.seq, payload: Data(message.payload))
  }

  func close() {
    transport.close()
  }
}

struct TestError: Error, CustomStringConvertible {
  let description: String

  init(_ description: String) {
    self.description = description
  }
}

/// A tiny model, its identity, and the frames and outputs Python recorded.
struct Golden {
  let name: String
  let model: URL
  let sha256: String
  let spec: [String: Any]
  let frames: Data
  let expected: Data

  init(_ name: String) throws {
    self.name = name
    model = Fixture.url("\(name).onnx")
    spec = try Fixture.json("\(name).spec.json")
    sha256 = spec["sha256"] as! String
    frames = try Fixture.data("\(name).frames.bin")
    expected = try Fixture.data("\(name).expected.bin")
  }
}
