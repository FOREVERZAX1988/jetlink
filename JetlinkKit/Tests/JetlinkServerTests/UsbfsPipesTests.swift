import Foundation
import JetlinkKit
import Testing

@testable import JetlinkServer

/// usbfs in memory, as the kernel behaves for one file descriptor: IN URBs
/// wait for what the comma sends, OUT URBs complete at once, a discarded URB
/// comes back with -ENOENT and whatever it had, and every finished URB waits
/// in one completion queue until it is reaped.
final class FakeUsbfs: UsbfsKernel, @unchecked Sendable {
  private let condition = NSCondition()
  private var inbound: [UInt8] = []
  private var pending: [UsbfsURB] = []
  private var completed: [UsbfsURB] = []
  private var woken = false
  private(set) var alive = true
  private(set) var written = Data()
  private(set) var discards = 0
  private(set) var reaps = 0
  /// An errno every submit fails with, when set.
  var failSubmit: Int32 = 0
  /// Most bytes one IN URB gets before it completes, as a burst on the bus.
  var burst = Int.max

  /// The comma sends `bytes`.
  func feed(_ bytes: [UInt8]) {
    condition.lock()
    inbound += bytes
    serve()
    condition.broadcast()
    condition.unlock()
  }

  /// The cable comes out: pending URBs die with -ESHUTDOWN, and the
  /// descriptor reports the device gone once they are reaped.
  func unplug() {
    condition.lock()
    alive = false
    for urb in pending {
      urb.status = -LinuxErrno.shutdown
      completed.append(urb)
    }
    pending.removeAll()
    condition.broadcast()
    condition.unlock()
  }

  var pendingCount: Int {
    condition.lock()
    defer { condition.unlock() }
    return pending.count
  }

  private func serve() {
    while let urb = pending.first, !inbound.isEmpty {
      let n = min(urb.count, inbound.count, burst)
      inbound.withUnsafeBytes { urb.buffer.copyMemory(from: $0.baseAddress!, byteCount: n) }
      inbound.removeFirst(n)
      urb.actual = n
      urb.status = 0
      completed.append(pending.removeFirst())
    }
  }

  func submit(_ urb: UsbfsURB) -> Int32 {
    condition.lock()
    defer { condition.unlock() }
    if failSubmit != 0 { return failSubmit }
    if !alive { return LinuxErrno.nodev }
    if urb.endpoint & 0x80 == 0 {
      written.append(urb.buffer.assumingMemoryBound(to: UInt8.self), count: urb.count)
      urb.actual = urb.count
      completed.append(urb)
    } else {
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
      reaps += 1
      return completed.removeFirst()
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
    condition.lock()
    woken = true
    condition.broadcast()
    condition.unlock()
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

@Suite("usbfs pipes")
struct UsbfsPipesTests {
  let kernel = FakeUsbfs()
  var device: UsbfsDevice { UsbfsDevice(kernel: kernel) }

  func pipes(_ device: UsbfsDevice) -> UsbfsPipes {
    UsbfsPipes(device: device, inEndpoint: 0x81, outEndpoint: 0x01)
  }

  @Test("A read returns what arrived, and a write sends it all in one URB")
  func readAndWrite() throws {
    let pipes = pipes(device)
    kernel.feed(Array(repeating: 7, count: 3000))
    let buffer = UnsafeMutableRawPointer.allocate(byteCount: 4096, alignment: 16)
    defer { buffer.deallocate() }
    #expect(try pipes.read(into: buffer, count: 4096, timeout: 0) == 3000)
    let bytes: [UInt8] = [1, 2, 3, 4, 5]
    #expect(try bytes.withUnsafeBytes { try pipes.write(from: $0.baseAddress!, count: 5, timeout: 2) } == 5)
    #expect(kernel.written == Data(bytes))
  }

  @Test("A read that times out returns what came before the deadline, not an error")
  func timeoutIsPartial() throws {
    kernel.burst = 1000
    let pipes = pipes(device)
    let buffer = UnsafeMutableRawPointer.allocate(byteCount: 4096, alignment: 16)
    defer { buffer.deallocate() }
    #expect(try pipes.read(into: buffer, count: 4096, timeout: 0.05) == 0)
    #expect(kernel.discards == 1)
    #expect(kernel.pendingCount == 0)
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

  @Test("invalidate ends every transfer and returns once none touches the descriptor")
  func invalidateWaitsForTransfers() throws {
    let device = device
    let pipes = pipes(device)
    let readers = (0..<2).map { _ in
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
    kernel.burst = 16 * 1024
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
}
