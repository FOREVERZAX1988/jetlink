import Foundation
import JetlinkKit
import Testing

@testable import JetlinkServer

/// The whole server, over a real socket, on onnxruntime's CPU provider: the
/// upload, the Swift preparation, the build, the load, the queues or the state
/// loop, and the reply, checked against what the Python server's parts
/// compute for the same frames (bit for bit on Apple; see `CommaClient.replay`).
@Suite("Server", .serialized)
struct ServerTests {
  func serve(_ body: (Server, TestClient) throws -> Void) throws {
    let cache = try TemporaryDirectory()
    let server = try Server(
      configuration: Server.Configuration(host: "127.0.0.1", port: 0, cacheRoot: cache.url, preload: false), backend: cpuBackend())
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
      let (hello, count) = try client.replay(golden)
      #expect(hello["protocol"] as? Int == 2)
      #expect(hello["backend"] as? String == "ort")
      #expect(count == 8)
      #expect(eventually { server.framesServed == count })
    }
  }

  @Test("The link names its medium: TCP until the comma's hello says it is a USB cable")
  func linkMedium() throws {
    let links = LockedLinks()
    try serve { server, client in
      server.host.subscribe { if case .link(let link) = $0 { links.append(link) } }
      try client.send(.ping)
      _ = try client.recv(.pong)
      #expect(server.currentLink.linkMedium == .tcp)
      let hello: [String: Any] = ["client": ["name": "modeld", "nonce": 1, "link": ["kind": "cable", "usb_speed": "high-speed"]]]
      try client.send(.helloReq, JSONSerialization.data(withJSONObject: hello))
      _ = try client.recv(.helloResp)
      #expect(server.currentLink.linkMedium == .usb2)
    }
    let media = links.all.filter { $0.state == .connected }.compactMap(\.linkMedium)
    #expect(media.last == .usb2)
    #expect(media.dropLast().allSatisfy { $0 == .tcp })
  }

  @Test("Progress is throttled within a stage, never across one")
  func progressStages() throws {
    try serve { server, _ in
      let seen = LockedLines()
      server.host.subscribe { if case .progress(let stage, _, let msg) = $0 { seen.append("\(stage) \(msg)") } }
      server.host.progress("patch", 0, "preparing")
      server.host.progress("patch", 0.5, "halfway")
      server.host.progress("compile", 0, "compiling")
      server.host.progress("compile", 0.1, "still compiling")
      server.host.progress("compile", 1, "compiled")
      #expect(seen.all == ["patch preparing", "compile compiling", "compile compiled"])
    }
  }

  @Test("A replayed request is dropped, and pings are answered")
  func dropsReplays() throws {
    try serve { _, client in
      let seq = try client.send(.ping)
      #expect(try client.recv().type == Wire.Msg.pong.rawValue)
      try client.send(.ping, seq: seq)  // a replay: no answer
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
      #expect(reply.status == Wire.Status.notReady.rawValue)
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
      #expect(try !FileManager.default.fileExists(atPath: server.cache.modelPath(golden.sha256).path))
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

  @Test("A benchmark runs the loaded engine at the comma's pace and reports in windows")
  func benchmarks() throws {
    let golden = try Golden("tiny_queued")
    try serve { server, client in
      _ = try client.ensureEngine(model: golden.model, sha256: golden.sha256)
      // Refused with a comma on the line.
      #expect(throws: HostError.self) { try server.host.benchmark(seconds: 1, run: BenchmarkRun()) }
      client.close()
      let deadline = Date().addingTimeInterval(5)
      while server.host.lock.withLock({ server.host.session != nil }) && Date() < deadline {
        Thread.sleep(forTimeInterval: 0.02)
      }
      let events = LockedEvents()
      server.host.subscribe { event in
        if case .benchmark(let value) = event { events.append(value) }
      }
      let report = try server.host.benchmark(seconds: 2, run: BenchmarkRun())
      #expect(report.sha256 == golden.sha256)
      #expect(report.device == server.backend.deviceTag())
      #expect(report.frames >= 30 && report.frames <= 45, "\(report.frames) frames in 2 s at 20 Hz")
      #expect(report.frame.mean > 0 && report.frame.p99 >= report.frame.p50 && report.frame.max >= report.frame.p99)
      #expect(report.windows.count == 1)
      #expect(report.windows[0].startSecond == 0)
      #expect(report.over50 <= report.over35)
      #expect(!report.cancelled)
      #expect(report.build.contains("CPU keep-warm off"))
      #expect(report.thermalAtStart != "")
      #expect(report.text.contains("frame        mean"))
      let seen = events.all
      #expect(seen.first?.state == "running")
      #expect(seen.last?.state == "done")
      #expect(seen.last?.report == report)
      #expect(!server.host.lock.withLock { server.host.benchmarking })
    }
  }

  @Test("A benchmark can be cancelled")
  func cancelsABenchmark() throws {
    let golden = try Golden("tiny_stateful")
    try serve { server, client in
      _ = try client.ensureEngine(model: golden.model, sha256: golden.sha256)
      client.close()
      let deadline = Date().addingTimeInterval(5)
      while server.host.lock.withLock({ server.host.session != nil }) && Date() < deadline {
        Thread.sleep(forTimeInterval: 0.02)
      }
      let run = BenchmarkRun()
      Thread {
        Thread.sleep(forTimeInterval: 0.6)
        run.cancel()
      }.start()
      let report = try server.host.benchmark(seconds: 60, run: run)
      #expect(report.cancelled)
      #expect(report.seconds < 5)
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

/// The listener beside the sessions: it heals, and stopping keeps the engine.
@Suite("Server lifecycle", .serialized)
struct ServerLifecycleTests {
  func makeServer(_ cache: TemporaryDirectory, dial: DialTarget? = nil) throws -> Server {
    try Server(
      configuration: Server.Configuration(host: "127.0.0.1", port: 0, cacheRoot: cache.url, preload: false, dial: dial), backend: cpuBackend())
  }

  @Test("stop keeps the engine loaded; shutdown releases it")
  func stopKeepsTheEngine() throws {
    let golden = try Golden("tiny_queued")
    let cache = try TemporaryDirectory()
    let server = try makeServer(cache)
    try server.start()
    let client = try TestClient(port: server.port!)
    _ = try client.ensureEngine(model: golden.model, sha256: golden.sha256)
    client.close()
    server.stop()
    #expect(server.host.loadedSHA() == golden.sha256)
    #expect(server.port == nil)
    server.shutdown()
    #expect(server.host.loadedSHA() == nil)
  }

  @Test("The log sink hears what the server logs")
  func logSink() throws {
    let lines = LockedLines()
    Log.sink = { level, category, message in lines.append("\(level.rawValue) \(category): \(message)") }
    defer { Log.sink = nil }
    let cache = try TemporaryDirectory()
    let server = try makeServer(cache)
    try server.start()
    server.shutdown()
    #expect(lines.all.contains { $0.hasPrefix("info server: listening on 127.0.0.1:") })
  }

  @Test("A listener that goes away is opened again")
  func listenerHeals() throws {
    let cache = try TemporaryDirectory()
    let server = try makeServer(cache)
    try server.start()
    let first = server.port!
    // The accept loop's listener ends as iOS ends it: closed under the loop.
    server.listener?.close()
    let deadline = Date().addingTimeInterval(5)
    while server.isListening && Date() < deadline {
      Thread.sleep(forTimeInterval: 0.05)
    }
    #expect(!server.isListening)
    while !server.isListening && Date() < deadline {
      Thread.sleep(forTimeInterval: 0.05)
    }
    guard let port = server.port else { throw TestError("the server did not listen again after \(first) went away") }
    let client = try TestClient(port: port)
    defer { client.close(); server.shutdown() }
    try client.send(.ping)
    #expect(try client.recv().type == Wire.Msg.pong.rawValue)
  }

  @Test("A dialed connection is served like an accepted one, and dialed again after it ends")
  func dials() throws {
    let cache = try TemporaryDirectory()
    // The comma's end: a listener the server dials.
    let comma = try TCPListener(host: "127.0.0.1", port: 0)
    let server = try makeServer(cache, dial: DialTarget(host: "127.0.0.1", port: comma.port))
    try server.start()
    defer { server.shutdown(); comma.close() }
    for round in 0..<2 {
      guard let transport = comma.accept() else { throw TestError("no dial in round \(round)") }
      transport.setReceiveTimeout(10)
      try transport.send(.ping, seq: UInt32(round + 1))
      let reply = try transport.recv()
      #expect(reply.msgType == Wire.Msg.pong.rawValue)
      #expect(reply.seq == UInt32(round + 1))
      transport.close()
    }
    // A listener keeps accepting beside the dialing.
    let client = try TestClient(port: server.port!)
    defer { client.close() }
    try client.send(.ping)
    #expect(try client.recv().type == Wire.Msg.pong.rawValue)
    server.setDial(nil)
    #expect(server.dialTarget == nil)
  }

  @Test("A dial target that does not answer is retried until it does")
  func dialsUntilAnswered() throws {
    let cache = try TemporaryDirectory()
    // A port with nobody on it, until the listener opens below.
    let probe = try TCPListener(host: "127.0.0.1", port: 0)
    let port = probe.port
    probe.close()
    let server = try makeServer(cache)
    try server.start()
    defer { server.shutdown() }
    server.setDial(DialTarget(host: "127.0.0.1", port: port))
    Thread.sleep(forTimeInterval: 0.7)
    let comma = try TCPListener(host: "127.0.0.1", port: port)
    defer { comma.close() }
    guard let transport = comma.accept() else { throw TestError("never dialed") }
    transport.setReceiveTimeout(10)
    try transport.send(.ping, seq: 1)
    #expect(try transport.recv().msgType == Wire.Msg.pong.rawValue)
    transport.close()
  }
}

final class LockedEvents: @unchecked Sendable {
  private let lock = NSLock()
  private var events: [BenchmarkEvent] = []

  func append(_ event: BenchmarkEvent) {
    lock.withLock { events.append(event) }
  }

  var all: [BenchmarkEvent] { lock.withLock { events } }
}

final class LockedLines: @unchecked Sendable {
  private let lock = NSLock()
  private var lines: [String] = []

  func append(_ line: String) {
    lock.withLock { lines.append(line) }
  }

  var all: [String] { lock.withLock { lines } }
}
