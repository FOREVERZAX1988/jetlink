import Foundation
import JetlinkKit
import Testing

@testable import JetlinkServer

/// The CPU backend with what a TensorRT one adds: fields in the hello, and
/// runs or loads that fail with an error of the test's choosing.
final class HookBackend: EngineBackend, @unchecked Sendable {
  private let inner = cpuBackend()
  private let lock = NSLock()
  private var failure: (any Error)?
  private var loadFailure: (any Error)?
  let helloFields: [String: Any]

  init(helloFields: [String: Any] = [:]) {
    self.helloFields = helloFields
  }

  var name: String { inner.name }
  var suffix: String { inner.suffix }
  var artifactKind: ArtifactKind { inner.artifactKind }
  var runtimeVersion: String { inner.runtimeVersion }
  func deviceTag() -> String { inner.deviceTag() }

  /// Every run from now on throws `error`; nil runs the model again.
  func failRuns(with error: (any Error)?) {
    lock.withLock { failure = error }
  }

  var runFailure: (any Error)? { lock.withLock { failure } }

  /// Every load from now on throws `error`; nil loads again.
  func failLoads(with error: (any Error)?) {
    lock.withLock { loadFailure = error }
  }

  func deriveSpec(model: URL, sha256: String, nbytes: Int64, frameSkip: Int) throws -> ModelSpec {
    try inner.deriveSpec(model: model, sha256: sha256, nbytes: nbytes, frameSkip: frameSkip)
  }

  func build(model: URL, artifact: URL, report: @escaping ProgressFn, metaExtra: [String: Any]) throws {
    try inner.build(model: model, artifact: artifact, report: report, metaExtra: metaExtra)
  }

  func load(artifact: URL, report: @escaping ProgressFn) throws -> any Engine {
    if let failure = lock.withLock({ loadFailure }) { throw failure }
    return FailingEngine(try inner.load(artifact: artifact, report: report), backend: self)
  }
}

/// The engine it loads: the real one, whose run throws when told to.
final class FailingEngine: Engine {
  let inner: any Engine
  unowned let backend: HookBackend

  init(_ inner: any Engine, backend: HookBackend) {
    self.inner = inner
    self.backend = backend
  }

  var inputs: [String: TensorSpec] { inner.inputs }
  var outputs: [String: TensorSpec] { inner.outputs }
  var hostOutputs: [String] { inner.hostOutputs }
  var lastGpuUs: UInt32 { inner.lastGpuUs }
  func hostInput(_ name: String) -> UnsafeMutableRawPointer? { inner.hostInput(name) }
  func output(_ name: String) -> UnsafeRawPointer? { inner.output(name) }
  func loopState(_ pairs: [(input: String, output: String)]) throws -> Bool { try inner.loopState(pairs) }
  func resetState() { inner.resetState() }
  func warm() throws -> String { try inner.warm() }
  func close() { inner.close() }

  func run() throws {
    if let failure = backend.runFailure { throw failure }
    try inner.run()
  }
}

/// A CUDA error as the TensorRT backend will throw it: sticky or not.
struct DeviceError: FatalEngineError, CustomStringConvertible {
  let sticky: Bool
  var isFatal: Bool { sticky }
  var description: String { sticky ? "CUDA_ERROR_ILLEGAL_ADDRESS" : "CUDA_ERROR_OUT_OF_MEMORY" }
}

/// A gadget that is off the bus until `plug` and back off after the comma
/// unplugs, recording when the USB loop looked.
final class ComingAndGoingGadget: GadgetSource, @unchecked Sendable {
  private let lock = NSLock()
  private var pipes: FakePipes?
  private var looked: [TimeInterval] = []

  func plug(_ pipes: FakePipes) {
    lock.withLock { self.pipes = pipes }
  }

  func unplug() {
    lock.withLock {
      pipes?.unplug()
      pipes = nil
    }
  }

  var looks: [TimeInterval] { lock.withLock { looked } }

  func present() -> Bool {
    lock.withLock {
      looked.append(ProcessInfo.processInfo.systemUptime)
      return pipes != nil
    }
  }

  func open() throws -> any MessageLink {
    guard let pipes = lock.withLock({ pipes }) else { throw LinkError.closed("no gadget") }
    return USBTransport(pipes: pipes, medium: .usb3)
  }
}

final class Recorded<T>: @unchecked Sendable {
  private let condition = NSCondition()
  private var values: [T] = []

  func append(_ value: T) {
    condition.lock()
    values.append(value)
    condition.broadcast()
    condition.unlock()
  }

  var all: [T] {
    condition.lock()
    defer { condition.unlock() }
    return values
  }

  /// Whether the values come to satisfy `done` within `timeout`. Each append
  /// wakes it, so the bound is only for a test that fails: a loaded CI
  /// runner can take seconds to get a thread to the event.
  func wait(timeout: TimeInterval = 10, until done: ([T]) -> Bool) -> Bool {
    let deadline = Date().addingTimeInterval(timeout)
    condition.lock()
    defer { condition.unlock() }
    while !done(values) {
      if !condition.wait(until: deadline) { return done(values) }
    }
    return true
  }
}

/// Each of `ServerHooks`, as the Linux daemon will use it, and the defaults
/// the apps keep.
@Suite("Server hooks", .serialized)
struct ServerHooksTests {
  func serve(
    hooks: ServerHooks, backend: any EngineBackend = cpuBackend(), gadget: (any GadgetSource)? = nil, _ body: (Server, TestClient) throws -> Void
  ) throws {
    let cache = try TemporaryDirectory()
    let server = try Server(
      configuration: Server.Configuration(host: "127.0.0.1", port: 0, cacheRoot: cache.url, preload: false, usb: gadget != nil),
      backend: backend, gadget: gadget, hooks: hooks)
    try server.start()
    defer { server.shutdown() }
    let client = try TestClient(port: server.port!)
    defer { client.close() }
    try body(server, client)
  }

  func hello(_ client: TestClient) throws -> [String: Any] {
    try client.hello()
  }

  /// One frame of `golden`'s model, zeros throughout.
  func frame(_ client: TestClient, _ golden: Golden, flags: Wire.Flag = []) throws -> Reply {
    let spec = try ModelSpec.from(golden.spec)
    var request = Data(count: Wire.inferReqSize)
    request.withUnsafeMutableBytes {
      $0.storeBytes(of: UInt32(1).littleEndian, as: UInt32.self)
      $0.storeBytes(of: flags.rawValue.littleEndian, toByteOffset: 4, as: UInt32.self)
    }
    request.append(Data(count: spec.warpedBytes + spec.packedBytes))
    try client.send(.inferReq, request)
    return try client.recv(.inferResp)
  }

  @Test("The hello carries Python's keys, the host's sleep_after and what the backend adds")
  func hello() throws {
    let python: Set = [
      "protocol", "backend", "runtime_version", "device", "engine_state", "loaded", "frames_served", "cached_models", "telemetry", "sleep_after",
    ]
    try serve(hooks: ServerHooks()) { _, client in
      let hello = try hello(client)
      #expect(Set(hello.keys) == python)
      #expect(hello["sleep_after"] as? Double == 0)
    }
    let trt = HookBackend(helloFields: ["trt_version": "10.3.0"])
    try serve(hooks: ServerHooks(sleepAfter: 900), backend: trt) { _, client in
      let hello = try hello(client)
      #expect(Set(hello.keys) == python.union(["trt_version"]))
      #expect(hello["trt_version"] as? String == "10.3.0")
      #expect(hello["sleep_after"] as? Double == 900)
    }
  }

  @Test("Telemetry goes out in the hello, the state reply, and a frame that asks for it")
  func telemetry() throws {
    let golden = try Golden("tiny_stateful")
    let hooks = ServerHooks(telemetry: { ["temp_c": 51.5, "power_w": 12.25] })
    func readings(_ object: Any?) -> [String: Double]? {
      (object as? [String: Any])?.compactMapValues { ($0 as? NSNumber)?.doubleValue }
    }
    try serve(hooks: hooks) { _, client in
      #expect(readings(try hello(client)["telemetry"]) == ["temp_c": 51.5, "power_w": 12.25])
      try client.send(.stateReq)
      let state = try client.recv(.stateResp).json
      #expect(state["temp_c"] as? Double == 51.5 && state["engine_state"] as? String == "none")
      _ = try client.ensureEngine(model: golden.model, sha256: golden.sha256)
      let plain = try frame(client, golden)
      let spec = try ModelSpec.from(golden.spec)
      #expect(plain.payload.count == spec.inferRespBytes)
      let asked = try frame(client, golden, flags: .wantState)
      #expect(asked.status == Wire.Status.ok.rawValue)
      try #require(asked.payload.count > spec.inferRespBytes)
      let piggyback = asked.payload[spec.inferRespBytes...]
      #expect(readings(try JSONSerialization.jsonObject(with: piggyback)) == ["temp_c": 51.5, "power_w": 12.25])
    }
  }

  @Test("A benchmark reports the host's thermal state")
  func thermal() throws {
    let golden = try Golden("tiny_stateful")
    try serve(hooks: ServerHooks(thermal: { "serious" })) { server, client in
      _ = try client.ensureEngine(model: golden.model, sha256: golden.sha256)
      client.close()
      #expect(eventually(timeout: 10) { server.host.lock.withLock { server.host.session == nil } })
      let report = try server.host.benchmark(seconds: 1, run: BenchmarkRun())
      #expect(report.thermalAtStart == "serious" && report.thermalAtEnd == "serious")
      #expect(report.windows.allSatisfy { $0.thermal == "serious" })
    }
  }

  /// The sleeper's input: touched while the gadget is on the bus and at each
  /// connection's start and end, idle only while it is gone, and a look for
  /// the gadget at once after a sleep.
  @Test("The gadget's comings and goings reach the host, and a wake looks for the comma at once")
  func gadgetIdle() throws {
    let gadget = ComingAndGoingGadget()
    let events = Recorded<GadgetIdleEvent>()
    let slept = Recorded<TimeInterval>()
    let hooks = ServerHooks(gadgetIdle: { event in
      events.append(event)
      guard event == .absent, slept.all.isEmpty else { return false }
      slept.append(ProcessInfo.processInfo.systemUptime)
      return true
    })
    let cache = try TemporaryDirectory()
    let server = try Server(
      configuration: Server.Configuration(host: "127.0.0.1", port: 0, cacheRoot: cache.url, preload: false, listen: false, usb: true),
      backend: cpuBackend(), gadget: gadget, hooks: hooks)
    try server.start()
    defer { server.shutdown() }
    #expect(events.wait { $0.filter { $0 == .absent }.count >= 3 })
    // The first idle slept: the next look came at once, not a poll later.
    let woke = try #require(slept.all.first)
    let next = try #require(gadget.looks.first { $0 > woke })
    #expect(next - woke < Server.usbPoll / 2, "looked again \(next - woke) s after waking")

    let comma = FakePipes()
    gadget.plug(comma)
    let client = GadgetClient(comma)
    _ = try client.hello(name: "modeld")
    gadget.unplug()
    #expect(events.wait { $0.last == .absent })
    let seen = events.all.drop { $0 == .absent }
    #expect(Array(seen.prefix(3)) == [.present, .connected, .disconnected], "\(seen)")
    #expect(seen.dropFirst(3).allSatisfy { $0 == .absent }, "\(seen)")
  }

  @Test("No idle while a comma is served over TCP, though the gadget is off the bus")
  func noIdleWhileServed() throws {
    let gadget = ComingAndGoingGadget()
    let events = Recorded<GadgetIdleEvent>()
    let hooks = ServerHooks(gadgetIdle: { event in
      events.append(event)
      return false
    })
    try serve(hooks: hooks, gadget: gadget) { _, client in
      try client.send(.ping)
      _ = try client.recv(.pong)
      #expect(events.wait { $0.contains(.connected) })
      let connected = events.all.count
      Thread.sleep(forTimeInterval: Server.usbPoll * 3)
      #expect(events.all.count == connected, "\(events.all)")
      client.close()
      #expect(events.wait { $0.last == .absent })
      #expect(events.all[connected...].first == .disconnected)
    }
  }

  @Test("Without a shutdown hook, or when it declines, the reply is ok:false and the app hears of it")
  func shutdownRefused() throws {
    for hooks in [ServerHooks(), ServerHooks(shutdown: { _ in nil })] {
      try serve(hooks: hooks) { server, client in
        let heard = Recorded<String>()
        server.host.subscribe { if case .shutdownRequested(let reason) = $0 { heard.append(reason) } }
        try client.sendJSON(.shutdownReq, ["reason": "car battery"])
        let reply = try client.recv(.shutdownResp).json
        #expect(reply["ok"] as? Bool == false)
        #expect(heard.wait { $0 == ["car battery"] })
      }
    }
  }

  @Test("A shutdown hook that accepts gets its reply ok:true written before its action runs")
  func shutdownAccepted() throws {
    let comma = FakePipes()
    let reasons = Recorded<String>()
    let repliedFirst = Recorded<Bool>()
    let hooks = ServerHooks(shutdown: { reason in
      reasons.append(reason)
      return {
        let frames = (try? HostFrames.parse(comma.written)) ?? []
        repliedFirst.append(frames.contains { $0.type == Wire.Msg.shutdownResp.rawValue })
      }
    })
    let cache = try TemporaryDirectory()
    let server = try Server(
      configuration: Server.Configuration(host: "127.0.0.1", port: 0, cacheRoot: cache.url, preload: false, listen: false, usb: true),
      backend: cpuBackend(), gadget: FakeGadget([comma], then: unservedPipes), hooks: hooks)
    try server.start()
    defer { server.shutdown() }
    let client = GadgetClient(comma)
    try client.sendJSON(.shutdownReq, ["reason": "car battery"])
    let reply = try client.recv(.shutdownResp).json
    #expect(reply["ok"] as? Bool == true)
    #expect(reply["detail"] as? String == "powering off")
    #expect(repliedFirst.wait { $0 == [true] })
    #expect(reasons.all == ["car battery"])
  }

  @Test("A fatal engine error is answered INFER_FAILED, then handed to the host; another is only answered")
  func fatal() throws {
    let golden = try Golden("tiny_stateful")
    let backend = HookBackend()
    let fatal = Recorded<String>()
    try serve(hooks: ServerHooks(fatal: { fatal.append(String(describing: $0)) }), backend: backend) { _, client in
      _ = try client.ensureEngine(model: golden.model, sha256: golden.sha256)
      #expect(try frame(client, golden).status == Wire.Status.ok.rawValue)
      backend.failRuns(with: DeviceError(sticky: false))
      #expect(try frame(client, golden).status == Wire.Status.inferFailed.rawValue)
      backend.failRuns(with: HostError.failed("not an engine fault"))
      #expect(try frame(client, golden).status == Wire.Status.inferFailed.rawValue)
      #expect(fatal.all.isEmpty)
      backend.failRuns(with: DeviceError(sticky: true))
      #expect(try frame(client, golden).status == Wire.Status.inferFailed.rawValue)
      #expect(fatal.wait { $0 == ["CUDA_ERROR_ILLEGAL_ADDRESS"] })
    }
  }

  @Test("A fatal engine error while preparing the engine fails the job, then is handed to the host; another only fails it")
  func fatalWhilePreparing() throws {
    let golden = try Golden("tiny_stateful")
    let backend = HookBackend()
    let fatal = Recorded<String>()
    try serve(hooks: ServerHooks(fatal: { fatal.append(String(describing: $0)) }), backend: backend) { _, client in
      backend.failLoads(with: DeviceError(sticky: false))
      #expect(throws: TestError.self) { try client.ensureEngine(model: golden.model, sha256: golden.sha256) }
      #expect(fatal.all.isEmpty)
      // Built by the first try: this one only loads.
      backend.failLoads(with: DeviceError(sticky: true))
      #expect(throws: TestError.self) { try client.ensureEngine(model: golden.model, sha256: golden.sha256) }
      #expect(fatal.wait { $0 == ["CUDA_ERROR_ILLEGAL_ADDRESS"] })
    }
  }
}
