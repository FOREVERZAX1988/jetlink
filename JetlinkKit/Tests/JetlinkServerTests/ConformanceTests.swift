import Foundation
import JetlinkKit
import JetlinkTestSupport
import Testing

@testable import JetlinkServer

#if canImport(Glibc)
  import Glibc
#elseif canImport(Android)
  import Android
#endif

/// What the Python does, as JetlinkKit/Scripts/make_conformance_fixtures.py
/// wrote it into Fixtures/conformance. docs/conformance.md has the story.
enum Conformance {
  static func url(_ name: String) -> URL { Fixture.url("conformance/\(name)") }
  static func data(_ name: String) throws -> Data { try Data(contentsOf: url(name)) }
  static func json(_ name: String) throws -> [String: Any] {
    try JSONSerialization.jsonObject(with: data(name)) as! [String: Any]
  }
}

func int(_ value: Any?) -> Int { (value as! NSNumber).intValue }
func double(_ value: Any?) -> Double { (value as! NSNumber).doubleValue }

/// "HELLO_REQ" as Swift spells a case: "helloReq".
func camel(_ name: String) -> String {
  let words = name.lowercased().split(separator: "_")
  return words.enumerated().map { $0.offset == 0 ? String($0.element) : $0.element.prefix(1).uppercased() + $0.element.dropFirst() }.joined()
}

/// A message of wire.json, with its payload made the way the generator made it.
struct WireMessage {
  let type: Wire.Msg
  let seq: UInt32
  let flags: Wire.Flag
  let parts: [Data]
  var payload: Data { parts.reduce(Data(), +) }

  init(_ d: [String: Any]) {
    type = Wire.Msg(rawValue: UInt16(int(d["type"])))!
    seq = UInt32(int(d["seq"]))
    flags = Wire.Flag(rawValue: UInt32(int(d["flags"])))
    if let json = d["json"] as? String {
      parts = [Data(json.utf8)]
    } else {
      let lengths = (d["parts"] as! [NSNumber]).map(\.intValue)
      let seq = Int(seq)
      // In steps: as one expression Linux's type checker gives up on it.
      let total: Int = lengths.reduce(0, +)
      var bytes = [UInt8](repeating: 0, count: total)
      for i in 0..<total {
        bytes[i] = UInt8((seq * 31 + i * 7) % 251)
      }
      let body = Data(bytes)
      var at = 0
      var parts: [Data] = []
      for n in lengths {
        parts.append(body.subdata(in: at..<at + n))
        at += n
      }
      self.parts = parts
    }
  }

  static func all() throws -> [WireMessage] {
    (try Conformance.json("wire.json")["messages"] as! [[String: Any]]).map(WireMessage.init)
  }

  func send(over link: any MessageLink) throws {
    try link.send(type, seq: seq, data: parts, flags: flags)
  }

  func matches(_ message: Message) -> Bool {
    message.msgType == type.rawValue && message.seq == seq && Data(message.payload) == payload
  }
}

@Suite("Conformance: the constants pinned to the Python")
struct PinnedConstantTests {
  @Test("The wire's constants are protocol.py's")
  func wire() {
    #expect(Wire.magic == Pinned.magic)
    #expect(Wire.version == Pinned.protocolVersion)
    #expect(Wire.headerSize == Pinned.headerSize)
    #expect(Wire.packetMultiple == Pinned.packetMultiple)
    #expect(Wire.gadgetTxAlign == Pinned.gadgetTxAlign)
    #expect(Wire.maxMessage == Pinned.maxMessage)
    #expect(Wire.inferReqSize == Pinned.inferReqSize)
    #expect(Wire.inferRespSize == Pinned.inferRespSize)
    #expect(Wire.defaultPort == Pinned.defaultPort)
  }

  @Test("Every message, flag and status has Python's number and name, and no more")
  func numbering() {
    for (name, value) in Pinned.messageTypes {
      let message = Wire.Msg(rawValue: value)
      #expect(message.map { String(describing: $0) } == camel(name), "message \(name) = \(value)")
    }
    #expect((0...UInt16(255)).compactMap(Wire.Msg.init(rawValue:)).count == Pinned.messageTypes.count)
    let flags: [String: Wire.Flag] = ["RESET_QUEUES": .resetQueues, "WANT_STATE": .wantState, "PADDED": .padded]
    #expect(Set(flags.keys) == Set(Pinned.flags.map(\.name)))
    for (name, value) in Pinned.flags {
      #expect(flags[name]?.rawValue == value, "flag \(name)")
    }
    for (name, value) in Pinned.statuses {
      #expect(Wire.Status(rawValue: value).map { String(describing: $0) } == camel(name), "status \(name) = \(value)")
    }
    #expect((0...UInt32(255)).compactMap(Wire.Status.init(rawValue:)).count == Pinned.statuses.count)
  }

  @Test("The model constants, the USB sizes and the slow frame are the Python's")
  func model() {
    #expect(ModelConstants.runFrequency == Pinned.modelRunFrequency)
    #expect(ModelConstants.contextFrequency == Pinned.modelContextFrequency)
    #expect(ModelConstants.defaultFrameSkip == Pinned.defaultFrameSkip)
    #expect(ModelConstants.chunk == Pinned.uploadChunk)
    #expect(Int(FrameStats.slowUs) == Pinned.slowFrameUs)
    #expect(USBTransport.packetSize == Pinned.usbMaxPacket)
    #expect(USBTransport.readChunk == Pinned.usbReadChunk)
  }

  @Test("Builds carry the Python's prepare version, on the Python's onnxruntime")
  func preparation() {
    #if canImport(Metal)
      #expect(CoreMLBackend.prepareVersion == Pinned.prepareVersion)
    #endif
    #expect(QNNBackend.prepareVersion == Pinned.prepareVersion)
    #expect(OrtRuntime.version == Pinned.onnxruntimeVersion)
  }
}

@Suite("Conformance: wire bytes against protocol.py and StreamTransport")
struct WireConformanceTests {
  @Test("Headers pack and unpack as protocol.pack_header does")
  func headers() throws {
    for h in try Conformance.json("wire.json")["headers"] as! [[String: Any]] {
      let header = Wire.Header(
        msgType: UInt16(int(h["msg_type"])), seq: UInt32(int(h["seq"])), flags: UInt32(int(h["flags"])), length: UInt32(int(h["length"])),
        reserved: (h["reserved"] as! NSNumber).uint64Value)
      var packed = [UInt8](repeating: 0xEE, count: Wire.headerSize)
      packed.withUnsafeMutableBytes { Wire.packHeader(header, into: $0.baseAddress!) }
      #expect(packed == hex(h["hex"] as! String), "\(h["name"]!)")
      let read = try hex(h["hex"] as! String).withUnsafeBytes { try Wire.unpackHeader($0.baseAddress!) }
      #expect(read == header, "\(h["name"]!)")
    }
  }

  @Test("INFER bodies are the bytes protocol.py packs")
  func inferBodies() throws {
    let wire = try Conformance.json("wire.json")
    for r in wire["infer_req"] as! [[String: Any]] {
      let raw = hex(r["hex"] as! String)
      raw.withUnsafeBytes {
        #expect(UInt32(littleEndian: $0.loadUnaligned(as: UInt32.self)) == UInt32(int(r["frame_id"])))
        #expect(UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: 4, as: UInt32.self)) == UInt32(int(r["flags"])))
      }
    }
    for r in wire["infer_resp"] as! [[String: Any]] {
      let packed = Wire.inferResp(
        frameID: UInt32(int(r["frame_id"])), status: Wire.Status(rawValue: UInt32(int(r["status"])))!, gpuUs: UInt32(int(r["gpu_us"])),
        queueUs: UInt32(int(r["queue_us"])), totalUs: UInt32(int(r["total_us"])))
      #expect(packed == hex(r["hex"] as! String))
    }
  }

  /// A connected pair of stream sockets: `a` for the transport, `b` for the test.
  private func socketPair() throws -> (Int32, Int32) {
    var fds: [Int32] = [0, 0]
    guard socketpair(AF_UNIX, Sys.stream, 0, &fds) == 0 else { throw TestError("socketpair failed") }
    return (fds[0], fds[1])
  }

  @Test("TCP sends what TcpTransport sends")
  func tcpSends() throws {
    let expected = try Conformance.data("wire.tcp.bin")
    let (a, b) = try socketPair()
    let transport = TCPTransport(fd: a, peer: "fixture")
    let reader = Thread.detachNewThreadAndCollect(fd: b, count: expected.count)
    for message in try WireMessage.all() {
      try message.send(over: transport)
    }
    let got = reader.wait()
    transport.close()
    close(b)
    #expect(got == expected)
  }

  @Test("TCP reads what TcpTransport sends")
  func tcpReads() throws {
    let stream = try Conformance.data("wire.tcp.bin")
    let (a, b) = try socketPair()
    let transport = TCPTransport(fd: a, peer: "fixture")
    let writer = Thread {
      stream.withUnsafeBytes { raw in
        var at = 0
        while at < raw.count {
          let n = write(b, raw.baseAddress! + at, raw.count - at)
          if n <= 0 { break }
          at += n
        }
      }
    }
    writer.start()
    for message in try WireMessage.all() {
      let got = try transport.recv()
      #expect(message.matches(got), "\(message.type) seq \(message.seq)")
    }
    transport.close()
    close(b)
  }

  @Test("A USB host sends what UsbBulkTransport sends")
  func usbHostSends() throws {
    let pipes = FakePipes()
    let transport = USBTransport(pipes: pipes)
    for message in try WireMessage.all() {
      try message.send(over: transport)
    }
    #expect(pipes.written == (try Conformance.data("wire.usb_host.bin")))
  }

  @Test("A USB host reads the gadget's bursts with the reads UsbBulkTransport posts")
  func usbHostReads() throws {
    let stream = try Conformance.data("wire.usb_gadget.bin")
    let pipes = FakePipes()
    pipes.push([UInt8](stream))
    let transport = USBTransport(pipes: pipes)
    for message in try WireMessage.all() {
      let got = try transport.recv()
      #expect(message.matches(got), "\(message.type) seq \(message.seq)")
    }
    let streams = try Conformance.json("wire.json")["streams"] as! [String: [String: Any]]
    let reads = (streams["usb_gadget"]!["reads"] as! [NSNumber]).map(\.intValue)
    #expect(pipes.readSizes == reads)
    #expect(pipes.drained)
  }
}

extension Thread {
  /// Reads `count` bytes from `fd` on a thread of its own; `wait()` returns them.
  static func detachNewThreadAndCollect(fd: Int32, count: Int) -> Collector {
    let collector = Collector()
    let thread = Thread {
      var out = Data()
      var buffer = [UInt8](repeating: 0, count: 1 << 16)
      while out.count < count {
        let n = buffer.withUnsafeMutableBytes { read(fd, $0.baseAddress!, $0.count) }
        if n <= 0 { break }
        out.append(contentsOf: buffer[0..<n])
      }
      collector.finish(out)
    }
    thread.start()
    return collector
  }

  final class Collector: @unchecked Sendable {
    private let condition = NSCondition()
    private var data: Data?

    func finish(_ value: Data) {
      condition.lock()
      data = value
      condition.broadcast()
      condition.unlock()
    }

    func wait() -> Data {
      condition.lock()
      defer { condition.unlock() }
      while data == nil { condition.wait() }
      return data!
    }
  }
}

/// An engine that is only its input buffers, for staging without a runtime.
final class StagingEngine: Engine {
  let inputs: [String: TensorSpec]
  let outputs: [String: TensorSpec] = [:]
  let lastGpuUs: UInt32 = 0
  private var buffers: [String: UnsafeMutableRawPointer] = [:]

  init(_ specs: [TensorSpec]) {
    inputs = Dictionary(uniqueKeysWithValues: specs.map { ($0.name, $0) })
    for spec in specs {
      let buffer = UnsafeMutableRawPointer.allocate(byteCount: spec.byteCount, alignment: 64)
      buffer.initializeMemory(as: UInt8.self, repeating: 0xAB, count: spec.byteCount)
      buffers[spec.name] = buffer
    }
  }

  deinit {
    for buffer in buffers.values { buffer.deallocate() }
  }

  func hostInput(_ name: String) -> UnsafeMutableRawPointer? { buffers[name] }
  func output(_ name: String) -> UnsafeRawPointer? { nil }
  func loopState(_ pairs: [(input: String, output: String)]) throws -> Bool { false }
  func resetState() {}
  func run() throws {}
  func warm() throws -> String { "" }
  func close() {}
}

@Suite("Conformance: queue staging against queues.PolicyQueues")
struct StagingConformanceTests {
  @Test("Each frame stages the tensors Python's queues feed", arguments: [1, 2, 4])
  func staging(frameSkip: Int) throws {
    let manifest = try Conformance.json("staging.json")
    let cases = manifest["cases"] as! [[String: Any]]
    let entry = try #require(cases.first { int($0["frame_skip"]) == frameSkip })
    let spec = try ModelSpec.from(Conformance.json(entry["spec"] as! String))
    #expect(spec.frameSkip == frameSkip)
    let inputs = (entry["inputs"] as! [[String: Any]]).map {
      TensorSpec(name: $0["name"] as! String, type: .float16, shape: ($0["shape"] as! [NSNumber]).map(\.intValue))
    }
    let engine = StagingEngine(inputs)
    let staging = try PolicyQueues(spec: spec, engine: engine)
    let frames = try Conformance.data(entry["frames"] as! String)
    let staged = try Conformance.data(entry["staged"] as! String)
    let frameBytes = spec.warpedBytes + spec.packedBytes
    let stagedBytes = inputs.reduce(0) { $0 + $1.byteCount }
    let count = int(manifest["frames"])
    #expect(frames.count == count * frameBytes)
    #expect(staged.count == count * stagedBytes)
    let packed = UnsafeMutableRawPointer.allocate(byteCount: spec.packedBytes, alignment: 16)
    defer { packed.deallocate() }
    for frame in 0..<count {
      if frame == int(manifest["reset_before"]) {
        staging.reset()
      }
      try frames.withUnsafeBytes { raw in
        let base = raw.baseAddress! + frame * frameBytes
        packed.copyMemory(from: base + spec.warpedBytes, byteCount: spec.packedBytes)
        try staging.stage(warped: base, packed: packed)
      }
      var offset = frame * stagedBytes
      for input in inputs {
        let got = Data(bytes: engine.hostInput(input.name)!, count: input.byteCount)
        #expect(got == staged.subdata(in: offset..<offset + input.byteCount), "frame \(frame) \(input.name) at frame_skip \(frameSkip)")
        offset += input.byteCount
      }
    }
  }
}

@Suite("Conformance: frame statistics against session.FrameStats")
struct StatsConformanceTests {
  @Test("The stats event is the one FrameStats.summary makes")
  func summaries() throws {
    let fixture = try Conformance.json("stats.json")
    #expect(int(fixture["slow_frame_us"]) == Int(FrameStats.slowUs))
    for c in fixture["cases"] as! [[String: Any]] {
      let name = c["name"] as! String
      let clock = Clock()
      let stats = FrameStats(now: clock.read)
      for sample in c["samples"] as! [[NSNumber]] {
        clock.value = sample[0].doubleValue
        stats.record(totalUs: sample[1].uint32Value, gpuUs: sample[2].uint32Value, queueUs: sample[3].uint32Value, sendUs: sample[4].uint32Value)
      }
      clock.value = double(c["now"])
      let summary = stats.summary(window: double(c["window"]), framesTotal: int(c["frames_total"]))
      guard let expected = c["expected"] as? [String: Any] else {
        #expect(summary == nil, "\(name)")
        continue
      }
      let got = try #require(summary, "\(name)")
      let served = expected["served_ms"] as! [String: Any]
      let stages = expected["stages_ms"] as! [String: Any]
      #expect(got.frames == int(expected["frames"]), "\(name)")
      #expect(got.fps == double(expected["fps"]), "\(name) fps")
      #expect(got.servedMs == StatsEvent.Total(mean: double(served["mean"]), p99: double(served["p99"]), max: double(served["max"])), "\(name) served")
      let wantStages = StatsEvent.Stages(
        queue: double(stages["queue"]), gpu: double(stages["gpu"]), other: double(stages["other"]), send: double(stages["send"]))
      #expect(got.stagesMs == wantStages, "\(name) stages")
      #expect(got.slow == int(expected["slow"]), "\(name) slow")
      #expect(got.windowS == double(expected["window_s"]), "\(name) window")
    }
  }

  final class Clock: @unchecked Sendable {
    var value: TimeInterval = 0
    var read: @Sendable () -> TimeInterval { { [unowned self] in self.value } }
  }
}
