import Foundation
import JetlinkKit
import JetlinkTestSupport
import Testing

@testable import JetlinkServer

/// usbfs in memory, as the kernel and the bus behave for one file descriptor.
/// What the comma writes arrives in 1024-byte packets. An IN URB fills with
/// them and completes when it is full or when a short packet (or a
/// zero-length one) ends one of the comma's writes; until then it waits,
/// holding what it has. OUT URBs complete at once. A discarded URB comes back
/// with -ENOENT and whatever it got. Every finished URB waits in one
/// completion queue until it is reaped.
final class FakeUsbfs: UsbfsKernel, @unchecked Sendable {
  static let packet = Pinned.usbMaxPacket
  private let condition = NSCondition()
  private var inbound: [UInt8] = []
  private var cursor = 0
  /// Offsets into `inbound` where a short packet ended a write.
  private var shortEnds: [Int] = []
  private var pending: [UsbfsURB] = []
  private var completed: [UsbfsURB] = []
  private var woken = false
  private(set) var alive = true
  private var outbound = Data()
  private(set) var discards = 0
  private(set) var submits = 0
  /// Every URB ever submitted, by identity.
  private(set) var urbs = Set<ObjectIdentifier>()
  /// An errno every submit fails with, when set.
  var failSubmit: Int32 = 0
  /// Most IN URBs in flight; a submit past it fails with ENOMEM, as usbfs's
  /// memory limit makes it.
  var maxPending = Int.max
  /// The status the next IN URB completes with, once, when set.
  var failNext: Int32 = 0
  /// Reaps hand finished URBs back in any order.
  var shuffleReaps = false

  /// The comma writes `bytes`: whole packets, then a short one when the
  /// length is not a multiple of a packet, or a zero-length one when empty.
  func feed(_ bytes: [UInt8]) {
    condition.withLock {
      inbound += bytes
      if bytes.count % FakeUsbfs.packet != 0 || bytes.isEmpty {
        shortEnds.append(inbound.count)
      }
      serve()
      condition.broadcast()
    }
  }

  /// The comma writes `bytes` as the bus carries them: a packet moves only
  /// into a posted IN URB, at `rate` bytes a second, as a device the host has
  /// not asked is NAKed. Returns when the first byte went on the wire, by
  /// `DispatchTime.now().uptimeNanoseconds`.
  func transmit(_ bytes: UnsafeRawBufferPointer, rate: Double) -> UInt64 {
    var sent = 0
    var first: UInt64 = 0
    while sent < bytes.count {
      condition.lock()
      while alive && !pending.contains(where: { $0.endpoint & 0x80 != 0 }) {
        condition.wait()
      }
      guard alive, let index = pending.firstIndex(where: { $0.endpoint & 0x80 != 0 }) else {
        condition.unlock()
        break
      }
      let urb = pending[index]
      let n = min(urb.count - urb.actual, bytes.count - sent, 16 * 1024)
      condition.unlock()
      // The bytes are on the wire for n / rate before the URB has them.
      let start = DispatchTime.now().uptimeNanoseconds
      if first == 0 { first = start }
      let end = start + UInt64(Double(n) / rate * 1e9)
      while DispatchTime.now().uptimeNanoseconds < end {}
      condition.lock()
      (urb.buffer + urb.actual).copyMemory(from: bytes.baseAddress! + sent, byteCount: n)
      urb.actual += n
      sent += n
      if urb.actual == urb.count, let index = pending.firstIndex(where: { $0 === urb }) {
        finish(pending.remove(at: index))
      }
      condition.broadcast()
      condition.unlock()
    }
    return first
  }

  /// The cable comes out: pending URBs die with -ESHUTDOWN, and the
  /// descriptor reports the device gone once they are reaped.
  func unplug() {
    condition.withLock {
      alive = false
      for urb in pending {
        urb.status = -LinuxErrno.shutdown
        completed.append(urb)
      }
      pending.removeAll()
      condition.broadcast()
    }
  }

  var pendingCount: Int { condition.withLock { pending.count } }
  /// The byte counts of the IN URBs in flight, oldest first.
  var pendingSizes: [Int] { condition.withLock { pending.filter { $0.endpoint & 0x80 != 0 }.map(\.count) } }
  /// What the comma sent that no URB has taken yet.
  var buffered: Int { condition.withLock { inbound.count - cursor } }
  var written: Data { condition.withLock { outbound } }

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

  private func finish(_ urb: UsbfsURB) {
    if failNext != 0 {
      urb.status = failNext
      failNext = 0
    }
    completed.append(urb)
  }

  private func serve() {
    while let index = pending.firstIndex(where: { $0.endpoint & 0x80 != 0 }) {
      let urb = pending[index]
      var n = min(urb.count - urb.actual, inbound.count - cursor)
      var ends = false
      if let short = shortEnds.first, short <= cursor + n {
        n = short - cursor
        ends = true
        shortEnds.removeFirst()
      }
      if n > 0 {
        inbound.withUnsafeBytes { (urb.buffer + urb.actual).copyMemory(from: $0.baseAddress! + cursor, byteCount: n) }
        urb.actual += n
        cursor += n
      }
      guard urb.actual == urb.count || ends else { break }
      finish(pending.remove(at: index))
    }
    if cursor > 1 << 20 {
      inbound.removeFirst(cursor)
      shortEnds = shortEnds.map { $0 - cursor }
      cursor = 0
    }
  }

  func submit(_ urb: UsbfsURB) -> Int32 {
    condition.lock()
    defer { condition.unlock() }
    if failSubmit != 0 { return failSubmit }
    if !alive { return LinuxErrno.nodev }
    submits += 1
    urbs.insert(ObjectIdentifier(urb))
    if urb.endpoint & 0x80 == 0 {
      outbound.append(urb.buffer.assumingMemoryBound(to: UInt8.self), count: urb.count)
      urb.actual = urb.count
      completed.append(urb)
    } else {
      if pending.count >= maxPending { return LinuxErrno.nomem }
      urb.actual = 0
      pending.append(urb)
      serve()
    }
    condition.broadcast()
    return 0
  }

  func discard(_ urb: UsbfsURB) {
    condition.lock()
    defer { condition.unlock() }
    guard let index = pending.firstIndex(where: { $0 === urb }) else { return }
    discards += 1
    pending.remove(at: index)
    urb.status = -LinuxErrno.noent
    completed.append(urb)
    condition.broadcast()
  }

  func reap() throws -> UsbfsURB? {
    condition.lock()
    defer { condition.unlock() }
    if !completed.isEmpty {
      return completed.remove(at: shuffleReaps ? Int.random(in: 0..<completed.count) : 0)
    }
    if !alive { throw LinkError.closed("the device is gone") }
    return nil
  }

  func wait(timeout: TimeInterval?) -> Bool {
    condition.lock()
    defer { condition.unlock() }
    let deadline = timeout.map { Date().addingTimeInterval($0) }
    while completed.isEmpty && alive && !woken {
      if let deadline {
        if !condition.wait(until: deadline) { break }
      } else {
        condition.wait()
      }
    }
    woken = false
    return alive || !completed.isEmpty
  }

  func wake() {
    condition.withLock {
      woken = true
      condition.broadcast()
    }
  }
}

extension FakeUsbfs: FakeGadgetEnd {
  func push(_ bytes: [UInt8]) {
    feed(bytes)
  }
}

/// Runs `body` on a thread of its own; `join` waits for its result.
final class Background<T>: @unchecked Sendable {
  private let done = DispatchSemaphore(value: 0)
  private var result: Result<T, any Error>?

  init(_ body: @escaping @Sendable () throws -> T) {
    let thread = Thread { [self] in
      result = Result { try body() }
      done.signal()
    }
    thread.start()
  }

  func join(timeout: TimeInterval = 10) -> Result<T, any Error>? {
    guard done.wait(timeout: .now() + timeout) == .success else { return nil }
    return result
  }
}

/// `count` bytes that differ from their neighbours, starting at `seed`.
func pattern(_ count: Int, seed: Int = 0) -> [UInt8] {
  (0..<count).map { UInt8(truncatingIfNeeded: ($0 + seed) &* 131 &+ ($0 + seed) >> 8) }
}

/// A scratch buffer for a read.
final class Scratch {
  let pointer: UnsafeMutableRawPointer
  let count: Int

  init(_ count: Int) {
    self.count = count
    pointer = .allocate(byteCount: count, alignment: 64)
  }

  deinit {
    pointer.deallocate()
  }

  func bytes(_ n: Int) -> [UInt8] {
    Array(UnsafeRawBufferPointer(start: pointer, count: n))
  }
}

@Suite("usbfs pipes")
struct UsbfsPipesTests {
  let kernel = FakeUsbfs()
  var device: UsbfsDevice { UsbfsDevice(kernel: kernel) }
  let slot = ReadRing.slotSize

  func pipes(_ device: UsbfsDevice, depth: Int = ReadRing.depth) -> UsbfsPipes {
    UsbfsPipes(device: device, inEndpoint: 0x81, outEndpoint: 0x01, depth: depth)
  }

  /// Reads until `count` bytes arrived.
  func readAll(_ pipes: UsbfsPipes, _ count: Int) throws -> [UInt8] {
    let scratch = Scratch(count)
    var got = 0
    while got < count {
      got += try pipes.read(into: scratch.pointer + got, count: count - got, timeout: 5)
    }
    return scratch.bytes(count)
  }

  @Test("A read returns what arrived, at most what it asked for, and a write sends it all in one URB")
  func readAndWrite() throws {
    let pipes = pipes(device)
    let sent = pattern(slot)
    kernel.feed(sent)
    let scratch = Scratch(slot)
    #expect(try pipes.read(into: scratch.pointer, count: 4096, timeout: 0) == 4096)
    #expect(try pipes.read(into: scratch.pointer + 4096, count: slot, timeout: 0) == slot - 4096)
    #expect(scratch.bytes(slot) == sent)
    let bytes: [UInt8] = [1, 2, 3, 4, 5]
    #expect(try bytes.withUnsafeBytes { try pipes.write(from: $0.baseAddress!, count: 5, timeout: 2) } == 5)
    #expect(kernel.written == Data(bytes))
  }

  @Test("The ring posts its reads before the comma sends, so a whole request lands with no read asked for in between")
  func ringIsPostedAhead() throws {
    let pipes = pipes(device)
    let scratch = Scratch(64)
    #expect(try pipes.read(into: scratch.pointer, count: 64, timeout: 0.01) == 0)
    #expect(kernel.pendingSizes == Array(repeating: slot, count: ReadRing.depth))
    // A 766 MB model's request, as the comma pads it: 29 slots.
    let request = pattern(475_136)
    kernel.feed(request)
    #expect(kernel.buffered == 0, "the comma waited for a read")
    #expect(try readAll(pipes, request.count) == request)
    #expect(kernel.discards == 0)
  }

  @Test("A read that times out returns 0 and leaves the ring posted, so what comes later is kept")
  func timeoutKeepsRing() throws {
    let pipes = pipes(device)
    let scratch = Scratch(slot)
    #expect(try pipes.read(into: scratch.pointer, count: slot, timeout: 0.05) == 0)
    #expect(kernel.discards == 0)
    #expect(kernel.pendingCount == ReadRing.depth)
    let sent = pattern(slot, seed: 3)
    kernel.feed(sent)
    #expect(try readAll(pipes, slot) == sent)
  }

  @Test("Bytes come out in the order the reads were posted, however the reaps are ordered, around the ring many times")
  func outOfOrderReaps() throws {
    kernel.shuffleReaps = true
    let pipes = pipes(device, depth: 4)
    var expected: [UInt8] = []
    var got: [UInt8] = []
    for i in 0..<40 {
      let chunk = pattern(slot * (1 + i % 7), seed: i)
      expected += chunk
      let reader = Background { [self] in try readAll(pipes, chunk.count) }
      kernel.feed(chunk)
      got += try #require(reader.join()).get()
    }
    #expect(got == expected)
    #expect(kernel.discards == 0)
  }

  @Test("After a short packet the ring ends the reads posted behind it, keeps their bytes in order, and posts only what is asked")
  func shortPacketRealigns() throws {
    let pipes = pipes(device)
    let scratch = Scratch(slot)
    #expect(try pipes.read(into: scratch.pointer, count: slot, timeout: 0.01) == 0)
    // A whole slot, then a write that ends on a short packet, then two and a
    // half slots with no end yet: the last of them would sit there, holding
    // its half, until the comma wrote again.
    let first = pattern(slot, seed: 1)
    let short = pattern(5000, seed: 2)
    let rest = pattern(slot * 5 / 2, seed: 3)
    kernel.feed(first)
    kernel.feed(short)
    kernel.feed(rest)
    let all = first + short + rest
    #expect(try readAll(pipes, all.count) == all)
    #expect(kernel.discards == ReadRing.depth - 4, "the half-full slot and the empty ones behind it")
    // Off the grid, only what the reader asks for is posted.
    let reader = Background { [self] in try readAll(pipes, 3072) }
    #expect(eventually { kernel.pendingSizes == [3072] })
    let next = pattern(3072, seed: 4)
    kernel.feed(next)
    #expect(try #require(reader.join()).get() == next)
  }

  @Test("A zero-length packet keeps the ring on the grid")
  func zeroLengthPacket() throws {
    let pipes = pipes(device)
    let scratch = Scratch(8)
    #expect(try pipes.read(into: scratch.pointer, count: 8, timeout: 0.01) == 0)
    kernel.feed([])
    let sent = pattern(slot * 2)
    kernel.feed(sent)
    #expect(try readAll(pipes, sent.count) == sent)
    #expect(kernel.discards == 0)
    _ = try pipes.read(into: scratch.pointer, count: 8, timeout: 0.01)
    #expect(kernel.pendingSizes == Array(repeating: slot, count: ReadRing.depth))
  }

  @Test("A bus error on a slot fails the read, and every read after")
  func slotFailure() throws {
    let pipes = pipes(device)
    kernel.failNext = -LinuxErrno.proto
    kernel.feed(pattern(slot))
    let scratch = Scratch(slot)
    #expect(throws: LinkError.self) { try pipes.read(into: scratch.pointer, count: slot, timeout: 0) }
    #expect(throws: LinkError.self) { try pipes.read(into: scratch.pointer, count: slot, timeout: 0) }
  }

  @Test("A submit refused while reads are posted leaves the ring shallower, not failed")
  func shallowRing() throws {
    kernel.maxPending = 3
    let pipes = pipes(device)
    let sent = pattern(slot * 10)
    let reader = Background { [self] in try readAll(pipes, sent.count) }
    #expect(eventually { kernel.pendingCount == 3 })
    kernel.feed(sent)
    #expect(try #require(reader.join()).get() == sent)
  }

  @Test("abort ends a read blocked without a deadline, from another thread")
  func abortWakesRead() throws {
    let pipes = pipes(device)
    let reader = Background {
      let buffer = UnsafeMutableRawPointer.allocate(byteCount: 1024, alignment: 16)
      defer { buffer.deallocate() }
      return try pipes.read(into: buffer, count: 1024, timeout: 0)
    }
    Thread.sleep(forTimeInterval: 0.05)
    pipes.abort()
    let result = try #require(reader.join())
    #expect(throws: LinkError.self) { try result.get() }
    #expect(kernel.pendingCount == 0)
    // and it stays aborted
    let buffer = UnsafeMutableRawPointer.allocate(byteCount: 1024, alignment: 16)
    defer { buffer.deallocate() }
    #expect(throws: LinkError.self) { try pipes.read(into: buffer, count: 1024, timeout: 0) }
  }

  @Test("A write completes while another thread waits in the kernel for a read")
  func oneReaperForBoth() throws {
    let pipes = pipes(device)
    let reader = Background {
      let buffer = UnsafeMutableRawPointer.allocate(byteCount: 1024, alignment: 16)
      defer { buffer.deallocate() }
      return try pipes.read(into: buffer, count: 1024, timeout: 0)
    }
    Thread.sleep(forTimeInterval: 0.05)
    let bytes = [UInt8](repeating: 9, count: 100)
    let writer = Background { try bytes.withUnsafeBytes { try pipes.write(from: $0.baseAddress!, count: 100, timeout: 2) } }
    #expect(try #require(writer.join()).get() == 100)
    kernel.feed([1, 2, 3])
    #expect(try #require(reader.join()).get() == 3)
  }

  @Test("Closing ends the ring's reads and waits for the kernel to give them back")
  func closeRetiresRing() throws {
    let pipes = pipes(device)
    let scratch = Scratch(8)
    #expect(try pipes.read(into: scratch.pointer, count: 8, timeout: 0.01) == 0)
    #expect(kernel.pendingCount == ReadRing.depth)
    pipes.close()
    #expect(kernel.pendingCount == 0)
    #expect(throws: LinkError.self) { try pipes.read(into: scratch.pointer, count: 8, timeout: 0) }
  }

  @Test("A new session's pipes on the same device start clean after the last one aborted")
  func pipesPerSession() throws {
    let device = device
    let first = pipes(device)
    first.abort()
    first.close()
    let second = pipes(device)
    kernel.feed([4, 5])
    let buffer = UnsafeMutableRawPointer.allocate(byteCount: 1024, alignment: 16)
    defer { buffer.deallocate() }
    #expect(try second.read(into: buffer, count: 1024, timeout: 0) == 2)
  }

  @Test("Unplugging fails a blocked read, and the device stays gone")
  func unplugFails() throws {
    let device = device
    let pipes = pipes(device)
    let reader = Background {
      let buffer = UnsafeMutableRawPointer.allocate(byteCount: 1024, alignment: 16)
      defer { buffer.deallocate() }
      return try pipes.read(into: buffer, count: 1024, timeout: 0)
    }
    Thread.sleep(forTimeInterval: 0.05)
    kernel.unplug()
    let result = try #require(reader.join())
    #expect(throws: LinkError.self) { try result.get() }
    #expect(device.isGone)
  }

  @Test("Unplugging with a message half in the ring fails the read, and closing does not wait on the dead device")
  func unplugMidRing() throws {
    let device = device
    let pipes = pipes(device)
    let transport = USBTransport(pipes: pipes)
    kernel.feed(Array(FakePipes.gadgetFrame(.inferReq, seq: 1, payload: Data(pattern(200_000))).prefix(slot * 5)))
    let reader = Background { try transport.recv() }
    // Five slots emptied and posted again: the reader waits for the rest.
    #expect(eventually { kernel.submits == ReadRing.depth + 5 })
    kernel.unplug()
    let result = try #require(reader.join())
    #expect(throws: LinkError.self) { try result.get() }
    #expect(device.isGone)
    let started = Date()
    transport.close()
    #expect(Date().timeIntervalSince(started) < 1)
  }

  @Test("invalidate ends every transfer and returns once none touches the descriptor")
  func invalidateWaitsForTransfers() throws {
    let device = device
    let sessions = [pipes(device), pipes(device)]
    let readers = sessions.map { pipes in
      Background {
        let buffer = UnsafeMutableRawPointer.allocate(byteCount: 1024, alignment: 16)
        defer { buffer.deallocate() }
        return try pipes.read(into: buffer, count: 1024, timeout: 0)
      }
    }
    Thread.sleep(forTimeInterval: 0.05)
    #expect(device.invalidate(timeout: 2))
    for reader in readers {
      #expect(throws: LinkError.self) { try #require(reader.join()).get() }
    }
    #expect(kernel.pendingCount == 0)
  }

  @Test("A submit the kernel refuses is a link error")
  func submitFailure() {
    kernel.failSubmit = LinuxErrno.nomem
    let pipes = pipes(device)
    let buffer = UnsafeMutableRawPointer.allocate(byteCount: 16, alignment: 16)
    defer { buffer.deallocate() }
    #expect(throws: LinkError.self) { try pipes.read(into: buffer, count: 16, timeout: 0) }
  }

  @Test("USBTransport frames messages over usbfs as it does over IOUSBHost")
  func transportOverUsbfs() throws {
    let transport = USBTransport(pipes: pipes(device))
    let payload = Data((0..<100_000).map { UInt8(truncatingIfNeeded: $0 &* 13) })
    kernel.feed(FakePipes.gadgetFrame(.inferReq, seq: 5, payload: payload))
    let message = try transport.recv()
    #expect(message.msgType == Wire.Msg.inferReq.rawValue)
    #expect(message.seq == 5)
    #expect(Data(message.payload) == payload)
    try transport.send(.pong, seq: 6)
    let frames = try HostFrames.parse(kernel.written)
    #expect(frames.count == 1 && frames[0].type == Wire.Msg.pong.rawValue && frames[0].seq == 6)
  }

  @Test("Messages queued back to back come out one at a time, each at the start of the buffer")
  func backToBack() throws {
    kernel.shuffleReaps = true
    let transport = USBTransport(pipes: pipes(device))
    let payloads = [0, 1, 16_352, 16_353, 475_104, 300_000, 4].map { Data(pattern($0, seed: $0)) }
    for (i, payload) in payloads.enumerated() {
      kernel.feed(FakePipes.gadgetFrame(.inferReq, seq: UInt32(i + 1), payload: payload))
    }
    var starts = Set<UnsafeRawPointer>()
    for (i, payload) in payloads.enumerated() {
      let message = try transport.recv()
      #expect(message.seq == UInt32(i + 1))
      #expect(Data(message.payload) == payload)
      #expect(Int(bitPattern: message.payload.baseAddress) % 16 == 0, "the payload is not aligned")
      starts.insert(message.payload.baseAddress! - Wire.headerSize)
    }
    #expect(starts.count == 1, "a message did not land at the start of the buffer")
    #expect(kernel.discards == 0)
  }

  @Test("A corrupt header latches the link desynced, drain empties what the comma still sends, and the next session reads cleanly")
  func desyncDrainReopen() throws {
    let device = device
    let first = USBTransport(pipes: pipes(device))
    kernel.feed([UInt8](repeating: 0xEE, count: slot))
    #expect(throws: LinkError.self) { try first.recv() }
    #expect(first.desynced)
    kernel.feed([UInt8](repeating: 0x11, count: 300 * 1024))
    first.drain(0.3)
    #expect(kernel.buffered == 0)
    // Drained already: close without the five seconds' drain it would add.
    first.shutdown()
    first.close()
    #expect(kernel.pendingCount == 0)
    let second = USBTransport(pipes: pipes(device))
    kernel.feed(FakePipes.gadgetFrame(.ping, seq: 1, payload: Data()))
    #expect(try second.recv().msgType == Wire.Msg.ping.rawValue)
  }

  @Test("The steady state reuses the pipes' URBs: none is made per transfer")
  func urbsArePooled() throws {
    let pipes = pipes(device)
    let transport = USBTransport(pipes: pipes)
    let payload = Data(pattern(475_104))
    let reply = Data(pattern(73_860))
    for seq in 1...100 {
      kernel.feed(FakePipes.gadgetFrame(.inferReq, seq: UInt32(seq), payload: payload))
      #expect(try transport.recv().seq == UInt32(seq))
      try reply.withUnsafeBytes { try transport.send(.inferResp, seq: UInt32(seq), parts: [$0]) }
    }
    #expect(kernel.urbs.count == ReadRing.depth + 1)
    #expect(kernel.submits > 100 * 29)
  }
}
