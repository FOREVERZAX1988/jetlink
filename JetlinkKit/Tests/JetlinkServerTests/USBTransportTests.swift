import Foundation
import JetlinkKit
import Testing

@testable import JetlinkServer

/// The comma's end of the bulk pair, in memory: what the gadget sends waits in
/// `inbound`, what the Mac writes lands in `written`. Reads return at most
/// `burst` bytes, as the bus delivers a large message in pieces.
final class FakePipes: BulkPipes, @unchecked Sendable {
  private let condition = NSCondition()
  private var inbound: [UInt8] = []
  private var ends: [Int] = []
  private var consumed = 0
  private var aborted = false
  private var failure: LinkError?
  private var outbound = Data()
  private var writeCount = 0
  private var requests: [Int] = []
  private var crossings = 0
  var burst = Int.max
  var writeLimit = Int.max
  var acceptsWrites = true

  /// One message as the gadget frames it: header, payload, and zeros to the
  /// next 16 KB, never the PADDED flag (`FfsTransport`, `tx_align`).
  static func gadgetFrame(_ type: Wire.Msg, seq: UInt32, payload: Data, flags: Wire.Flag = []) -> [UInt8] {
    var bytes = [UInt8](repeating: 0, count: Wire.headerSize)
    bytes.withUnsafeMutableBytes {
      Wire.packHeader(Wire.Header(msgType: type.rawValue, seq: seq, flags: flags.rawValue, length: UInt32(payload.count)), into: $0.baseAddress!)
    }
    bytes += payload
    bytes += [UInt8](repeating: 0, count: USBTransport.gadgetPad(bytes.count))
    return bytes
  }

  func push(_ bytes: [UInt8]) {
    condition.lock()
    inbound += bytes
    ends.append(inbound.count)
    condition.broadcast()
    condition.unlock()
  }

  func push(_ type: Wire.Msg, seq: UInt32, payload: Data = Data(), flags: Wire.Flag = []) {
    push(FakePipes.gadgetFrame(type, seq: seq, payload: payload, flags: flags))
  }

  /// The cable comes out: every read from now on fails.
  func unplug() {
    condition.lock()
    failure = .closed("the gadget went away")
    condition.broadcast()
    condition.unlock()
  }

  func read(into buffer: UnsafeMutableRawPointer, count: Int, timeout: TimeInterval) throws -> Int {
    condition.lock()
    defer { condition.unlock() }
    requests.append(count)
    let deadline = timeout > 0 ? Date().addingTimeInterval(timeout) : nil
    while consumed == inbound.count && !aborted && failure == nil {
      if let deadline {
        if !condition.wait(until: deadline) { return 0 }
      } else {
        condition.wait()
      }
    }
    if aborted { throw LinkError.closed("link closed") }
    if let failure, consumed == inbound.count { throw failure }
    let n = min(count, inbound.count - consumed, burst)
    if ends.contains(where: { $0 > consumed && $0 < consumed + n }) {
      crossings += 1
    }
    inbound.withUnsafeBytes { buffer.copyMemory(from: $0.baseAddress! + consumed, byteCount: n) }
    consumed += n
    return n
  }

  func write(from buffer: UnsafeRawPointer, count: Int, timeout: TimeInterval) throws -> Int {
    condition.lock()
    defer { condition.unlock() }
    if aborted { throw LinkError.closed("link closed") }
    writeCount += 1
    guard acceptsWrites else { return 0 }
    let n = min(count, writeLimit)
    outbound.append(buffer.assumingMemoryBound(to: UInt8.self), count: n)
    condition.broadcast()
    return n
  }

  func abort() {
    condition.lock()
    aborted = true
    condition.broadcast()
    condition.unlock()
  }

  func close() {
    abort()
  }

  var written: Data { condition.withLock { outbound } }
  var writes: Int { condition.withLock { writeCount } }
  var readSizes: [Int] { condition.withLock { requests } }
  /// Reads that returned the end of one message and the start of the next.
  var crossed: Int { condition.withLock { crossings } }
  var drained: Bool { condition.withLock { consumed == inbound.count } }

  /// Waits until `written` holds at least `count` bytes.
  func waitForWritten(_ count: Int, timeout: TimeInterval = 30) -> Data? {
    condition.lock()
    defer { condition.unlock() }
    let deadline = Date().addingTimeInterval(timeout)
    while outbound.count < count {
      if !condition.wait(until: deadline) { return nil }
    }
    return outbound
  }
}

/// What the Mac sent, message by message: the host keeps the one-byte PADDED rule.
struct HostFrames {
  struct Frame {
    let type: UInt16
    let seq: UInt32
    let flags: UInt32
    let payload: Data
  }

  static func parse(_ data: Data) throws -> [Frame] {
    var frames: [Frame] = []
    var offset = 0
    while offset + Wire.headerSize <= data.count {
      let header = try data.withUnsafeBytes { try Wire.unpackHeader($0.baseAddress! + offset) }
      let pad = Wire.Flag(rawValue: header.flags).contains(.padded) ? 1 : 0
      let end = offset + Wire.headerSize + Int(header.length)
      guard end + pad <= data.count else { break }
      frames.append(Frame(type: header.msgType, seq: header.seq, flags: header.flags, payload: data.subdata(in: (offset + Wire.headerSize)..<end)))
      offset = end + pad
    }
    return frames
  }
}

@Suite("USB framing")
struct USBTransportTests {
  @Test("Messages framed as the gadget sends them come out whole, and no read crosses a message's end")
  func receivesGadgetFrames() throws {
    let pipes = FakePipes()
    pipes.burst = 7 * 1024
    let transport = USBTransport(pipes: pipes)
    let payloads: [(Wire.Msg, Data)] = [
      (.helloReq, Data(#"{"client":{"name":"modeld","nonce":7}}"#.utf8)),
      (.inferReq, Data((0..<(8 + 393_216 + 65_536)).map { UInt8(truncatingIfNeeded: $0 &* 31) })),
      (.ping, Data()),
      // a body that is exactly one burst, so the gadget adds no pad
      (.uploadChunk, Data(repeating: 0xA5, count: Wire.gadgetTxAlign - Wire.headerSize)),
      // one byte over, so the pad is nearly a whole burst
      (.uploadChunk, Data(repeating: 0x5A, count: Wire.gadgetTxAlign - Wire.headerSize + 1)),
    ]
    for (i, (type, payload)) in payloads.enumerated() {
      pipes.push(type, seq: UInt32(i + 1), payload: payload)
    }
    for (i, (type, payload)) in payloads.enumerated() {
      let message = try transport.recv()
      #expect(message.msgType == type.rawValue)
      #expect(message.seq == UInt32(i + 1))
      #expect(Data(message.payload) == payload)
      #expect(Int(bitPattern: message.payload.baseAddress) % 16 == 0, "the payload is not aligned")
    }
    #expect(pipes.crossed == 0)
    #expect(pipes.drained)
    for size in pipes.readSizes {
      #expect(size > 0 && size % USBTransport.packetSize == 0 && size <= USBTransport.readChunk, "read of \(size) bytes")
    }
  }

  @Test("The read size is the rest of the message in whole packets, within the room and the chunk")
  func readSizes() {
    #expect(USBTransport.readSize(missing: 32, room: 1 << 20) == 1024)
    #expect(USBTransport.readSize(missing: 16_384 - 1024, room: 1 << 20) == 15_360)
    #expect(USBTransport.readSize(missing: 460_800, room: 1 << 20) == USBTransport.readChunk)
    #expect(USBTransport.readSize(missing: 5000, room: 3000) == 2048)
    #expect(USBTransport.readSize(missing: 5000, room: 1000) == 0)
    #expect(USBTransport.gadgetPad(32) == 16_352)
    #expect(USBTransport.gadgetPad(16_384) == 0)
    #expect(USBTransport.gadgetPad(16_385) == 16_383)
  }

  @Test("What the Mac sends keeps the one-byte PADDED rule, in one transfer a message")
  func sendsHostFrames() throws {
    let pipes = FakePipes()
    let transport = USBTransport(pipes: pipes)
    let exact = Data(repeating: 1, count: Wire.packetMultiple - Wire.headerSize)
    try transport.send(.pong, seq: 3)
    try exact.withUnsafeBytes { try transport.send(.inferResp, seq: 4, parts: [$0]) }
    #expect(pipes.writes == 2)
    let frames = try HostFrames.parse(pipes.written)
    #expect(frames.count == 2)
    #expect(frames[0].type == Wire.Msg.pong.rawValue && frames[0].payload.isEmpty && frames[0].flags == 0)
    #expect(frames[1].payload == exact)
    #expect(Wire.Flag(rawValue: frames[1].flags).contains(.padded))
    #expect(pipes.written.count == Wire.headerSize + Wire.headerSize + exact.count + 1)
  }

  @Test("A write the bus takes in pieces still sends the message once, whole")
  func partialWrites() throws {
    let pipes = FakePipes()
    pipes.writeLimit = 1000
    let transport = USBTransport(pipes: pipes)
    let payload = Data((0..<74_000).map { UInt8(truncatingIfNeeded: $0) })
    try payload.withUnsafeBytes { try transport.send(.inferResp, seq: 9, parts: [$0]) }
    let frames = try HostFrames.parse(pipes.written)
    #expect(frames.count == 1)
    #expect(frames.first?.payload == payload)
    #expect(pipes.writes > 1)
  }

  @Test("A write that moves nothing is a lost link")
  func stalledWrite() {
    let pipes = FakePipes()
    pipes.acceptsWrites = false
    let transport = USBTransport(pipes: pipes)
    #expect(throws: LinkError.self) { try transport.send(.pong, seq: 1) }
  }

  @Test("A corrupt header latches the link desynced, and drain swallows what the comma still sends")
  func desyncAndDrain() throws {
    let pipes = FakePipes()
    let transport = USBTransport(pipes: pipes)
    pipes.push([UInt8](repeating: 0xEE, count: Wire.gadgetTxAlign))
    #expect(throws: LinkError.self) { try transport.recv() }
    #expect(transport.desynced)
    #expect(throws: LinkError.self) { try transport.recv() }
    pipes.push([UInt8](repeating: 0x11, count: 300_000))
    let started = Date()
    transport.drain(0.3)
    #expect(pipes.drained)
    #expect(Date().timeIntervalSince(started) < 5)
  }

  @Test("A message bigger than the receive buffer grows it")
  func growsForBigMessages() throws {
    let pipes = FakePipes()
    let transport = USBTransport(pipes: pipes)
    let payload = Data((0..<(4 << 20) + 8).map { UInt8(truncatingIfNeeded: $0 &* 7) })
    pipes.push(.uploadChunk, seq: 1, payload: payload)
    pipes.push(.ping, seq: 2)
    #expect(Data(try transport.recv().payload) == payload)
    #expect(try transport.recv().msgType == Wire.Msg.ping.rawValue)
    #expect(pipes.crossed == 0)
  }

  @Test("shutdown wakes a receive blocked on another thread")
  func shutdownWakesRecv() throws {
    let pipes = FakePipes()
    let transport = USBTransport(pipes: pipes)
    let failed = Latch()
    let thread = Thread {
      if (try? transport.recv()) == nil { failed.release() }
    }
    thread.start()
    Thread.sleep(forTimeInterval: 0.1)
    transport.shutdown()
    failed.wait()
  }
}

/// The comma's gadget for the server's USB loop: a queue of pipe pairs, one
/// per open, and a count of opens.
final class FakeGadget: GadgetSource, @unchecked Sendable {
  private let lock = NSLock()
  private var pending: [FakePipes]
  private var opened = 0
  private let make: () -> FakePipes

  init(_ pipes: [FakePipes] = [], then make: @escaping () -> FakePipes) {
    pending = pipes
    self.make = make
  }

  var opens: Int { lock.withLock { opened } }

  func present() -> Bool { true }

  func open() throws -> any MessageLink {
    let pipes: FakePipes = lock.withLock {
      opened += 1
      return pending.isEmpty ? make() : pending.removeFirst()
    }
    return USBTransport(pipes: pipes, medium: .usb3)
  }

}

/// A gadget on the bus that nothing on the comma serves yet: every read fails
/// at once, as the endpoints do before a run borrows the link.
func unservedPipes() -> FakePipes {
  let pipes = FakePipes()
  pipes.unplug()
  return pipes
}

final class LockedLinks: @unchecked Sendable {
  private let lock = NSLock()
  private var links: [LinkEvent] = []

  func append(_ link: LinkEvent) {
    lock.withLock { links.append(link) }
  }

  var all: [LinkEvent] { lock.withLock { links } }
}

/// The comma over the fake USB pipes.
final class GadgetClient: CommaClient {
  let pipes: FakePipes
  private var seq: UInt32 = 0
  private var seen = 0

  init(_ pipes: FakePipes) {
    self.pipes = pipes
  }

  func sendMessage(_ type: Wire.Msg, _ payload: Data, flags: Wire.Flag, seq explicit: UInt32?) throws -> UInt32 {
    if explicit == nil { seq += 1 }
    let seq = explicit ?? self.seq
    pipes.push(type, seq: seq, payload: payload, flags: flags)
    return seq
  }

  /// The next message the server wrote to the pipes, waiting up to a minute.
  func recv() throws -> Reply {
    let deadline = Date().addingTimeInterval(60)
    while Date() < deadline {
      let frames = try HostFrames.parse(pipes.written)
      if seen < frames.count {
        let frame = frames[seen]
        seen += 1
        return Reply(type: frame.type, seq: frame.seq, payload: frame.payload)
      }
      _ = pipes.waitForWritten(pipes.written.count + 1, timeout: 0.2)
    }
    throw TestError("nothing from the server in 60 s")
  }
}

@Suite("Server over USB", .serialized)
struct ServerUSBTests {
  func makeServer(_ cache: TemporaryDirectory, gadget: FakeGadget) throws -> Server {
    let server = try Server(
      configuration: Server.Configuration(host: "127.0.0.1", port: 0, cacheRoot: cache.url, preload: false, listen: false, usb: true),
      backend: cpuBackend())
    server.gadget = gadget
    return server
  }

  @Test("A gadget nothing on the comma serves is retried quietly, with no link events")
  func unservedGadget() throws {
    let cache = try TemporaryDirectory()
    let gadget = FakeGadget(then: unservedPipes)
    let server = try makeServer(cache, gadget: gadget)
    let links = LockedLinks()
    server.host.subscribe { if case .link(let link) = $0 { links.append(link) } }
    try server.start()
    defer { server.shutdown() }
    #expect(server.port == nil, "a USB server opens no port")
    let deadline = Date().addingTimeInterval(10)
    while gadget.opens < 3 && Date() < deadline {
      Thread.sleep(forTimeInterval: 0.05)
    }
    #expect(gadget.opens >= 3)
    #expect(links.all.allSatisfy { $0.state == .waiting }, "\(links.all)")
  }

  @Test("The comma over USB is served as over TCP; the link is up on its first message and down when it goes")
  func servesOverUSB() throws {
    let cache = try TemporaryDirectory()
    let comma = FakePipes()
    let gadget = FakeGadget([unservedPipes(), comma], then: unservedPipes)
    let server = try makeServer(cache, gadget: gadget)
    let links = LockedLinks()
    server.host.subscribe { if case .link(let link) = $0 { links.append(link) } }
    try server.start()
    defer { server.shutdown() }
    let client = GadgetClient(comma)
    try client.sendJSON(.helloReq, ["client": ["name": "modeld", "nonce": 1]])
    let hello = try client.recv(.helloResp).json
    #expect(hello["protocol"] as? Int == Int(Wire.version))
    try client.send(.ping)
    _ = try client.recv(.pong)
    #expect(links.all.contains { $0.state == .connected && $0.peer == "usb" && $0.linkMedium == .usb3 })
    comma.unplug()
    let deadline = Date().addingTimeInterval(5)
    while !links.all.contains(where: { $0.state == .disconnected }) && Date() < deadline {
      Thread.sleep(forTimeInterval: 0.05)
    }
    #expect(links.all.filter { $0.state == .connected }.count == 1)
    #expect(links.all.contains { $0.state == .disconnected })
  }

  @Test("A comma over USB is served the outputs the Python server computes", arguments: ["tiny_queued", "tiny_stateful"])
  func servesGoldenFramesOverUSB(_ name: String) throws {
    let golden = try Golden(name)
    let cache = try TemporaryDirectory()
    let comma = FakePipes()
    comma.burst = 5 * 1024
    let server = try makeServer(cache, gadget: FakeGadget([comma], then: unservedPipes))
    try server.start()
    defer { server.shutdown() }
    let (_, count) = try GadgetClient(comma).replay(golden)
    #expect(comma.crossed == 0)
    #expect(eventually { server.framesServed == count })
  }
}
