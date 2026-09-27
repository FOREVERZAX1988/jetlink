import Foundation

/// The two bulk endpoints of the comma's vendor interface: IOUSBHost on a Mac,
/// a fake in the tests. Both calls block.
protocol BulkPipes: AnyObject, Sendable {
  /// Reads up to `count` bytes, a whole number of packets, into `buffer`, and
  /// returns how many arrived. A timeout is not an error: whatever did arrive
  /// is returned, because dropping it desyncs the stream. A `timeout` of 0
  /// waits until data comes or the link goes. Throws `LinkError` when the
  /// link is gone or the transfer failed.
  func read(into buffer: UnsafeMutableRawPointer, count: Int, timeout: TimeInterval) throws -> Int
  /// Writes `count` bytes as one transfer and returns how many went out, also
  /// after a timeout, so the caller resends only the rest.
  func write(from buffer: UnsafeRawPointer, count: Int, timeout: TimeInterval) throws -> Int
  /// Ends any transfer in flight with an error, from another thread.
  func abort()
  /// Aborts and lets the interface go.
  func close()
}

/// Where the server finds the comma's gadget: IOKit on a Mac, a fake in the
/// tests.
protocol GadgetSource: Sendable {
  /// Is the gadget on the bus? Cheap enough to poll twice a second.
  func present() -> Bool
  /// Opens the link interface's bulk pair.
  func open() throws -> USBTransport
  /// The bus speed it enumerated at, as a phrase for the log.
  func speed() -> String?
}

/// Framing over USB bulk transfers, the host's end: the Swift form of
/// `UsbBulkTransport` on `StreamTransport`, with the rules the bench taught.
///
/// - The gadget pads every message to `Wire.gadgetTxAlign` (16 KB), so each
///   read asks for exactly the rest of the current message, rounded up to a
///   whole packet, and never stays outstanding past its end. Reading further
///   desynced about once in 400 frames.
/// - A read asks for whole packets only (a bulk IN whose buffer is not a
///   packet multiple can overflow) and at most `readChunk` at a time, with a
///   packet of slack past the message so a grown buffer never ends with room
///   for zero packets.
/// - What this end sends keeps the one-byte PADDED rule, and goes out as one
///   transfer: libusb and IOUSBHost have no vectored bulk write, and several
///   writes let the host scheduler interleave them.
/// - Reads wait without a deadline, as the Python server's do between
///   messages; `shutdown()` aborts them. Writes give up after
///   `writeTimeout` without progress.
/// - After a protocol error nothing resynchronises the stream: the link is
///   latched desynced, and `drain` swallows what the comma is still sending
///   so its own frame timeout fires instead of a half-written message
///   blocking it.
///
/// Every message lands at the start of the receive buffer, so the float32
/// arrays inside an INFER stay aligned, and the steady state allocates nothing.
final class USBTransport: MessageLink, @unchecked Sendable {
  static let packetSize = Wire.packetMultiple
  static let readChunk = 256 * packetSize
  static let readSlack = packetSize
  static let writeTimeout: TimeInterval = 2

  let peer: String
  private let pipes: any BulkPipes
  private var rx: UnsafeMutableRawPointer
  private var capacity: Int
  private var start = 0
  private var end = 0
  private(set) var desynced = false
  private let sendLock = NSLock()
  private var tx: UnsafeMutableRawPointer
  private var txCapacity: Int

  init(pipes: any BulkPipes, peer: String = "usb") {
    self.pipes = pipes
    self.peer = peer
    capacity = 2 << 20
    rx = UnsafeMutableRawPointer.allocate(byteCount: capacity, alignment: 64)
    txCapacity = 1 << 20
    tx = UnsafeMutableRawPointer.allocate(byteCount: txCapacity, alignment: 64)
  }

  deinit {
    pipes.close()
    rx.deallocate()
    tx.deallocate()
  }

  // MARK: receiving

  func recv() throws -> Message {
    if desynced {
      throw LinkError.desynced("stream desynced; the link must be reopened")
    }
    try fill(Wire.headerSize)
    let header: Wire.Header
    do {
      header = try Wire.unpackHeader(rx + start)
      if Int(header.length) > Wire.maxMessage {
        throw Wire.ProtocolError.tooLong(header.length)
      }
    } catch let error as Wire.ProtocolError {
      desynced = true
      throw LinkError.desynced("protocol error, link unusable: \(error)")
    }
    let body = Wire.headerSize + Int(header.length)
    let total = body + USBTransport.gadgetPad(body)
    try fill(total)
    let payload = UnsafeRawBufferPointer(start: rx + start + Wire.headerSize, count: Int(header.length))
    start += total
    if start == end {
      start = 0
      end = 0
    }
    return Message(msgType: header.msgType, seq: header.seq, flags: header.flags, payload: payload)
  }

  /// The pad the gadget put after a message of `body` bytes.
  static func gadgetPad(_ body: Int) -> Int {
    (Wire.gadgetTxAlign - body % Wire.gadgetTxAlign) % Wire.gadgetTxAlign
  }

  /// The bytes to ask for when `missing` of the current message are still to
  /// come: whole packets, capped by the room left and by `readChunk`.
  static func readSize(missing: Int, room: Int) -> Int {
    let wanted = (missing + packetSize - 1) / packetSize * packetSize
    return min(wanted, room, readChunk) / packetSize * packetSize
  }

  /// Reads until `need` bytes of the current message are buffered.
  private func fill(_ need: Int) throws {
    reserve(need + USBTransport.readSlack)
    while end - start < need {
      let size = USBTransport.readSize(missing: need - (end - start), room: capacity - end)
      if size == 0 {
        // Every read would return nothing and this loop would spin while the
        // comma blocks. Say so rather than hang.
        throw LinkError.closed("no room to read the rest of a \(need) byte message (\(end - start) in hand)")
      }
      end += try pipes.read(into: rx + end, count: size, timeout: 0)
    }
  }

  private func reserve(_ need: Int) {
    if start > 0 && start + need > capacity {
      memmove(rx, rx + start, end - start)
      end -= start
      start = 0
    }
    if need > capacity {
      let grown = max(need, capacity * 2)
      let buffer = UnsafeMutableRawPointer.allocate(byteCount: grown, alignment: 64)
      buffer.copyMemory(from: rx + start, byteCount: end - start)
      rx.deallocate()
      rx = buffer
      capacity = grown
      end -= start
      start = 0
    }
  }

  /// After a desync, reads and drops what the comma is still sending until it
  /// goes quiet for `timeout` or the link drops. Reopening instead would
  /// drain a packet per session, and the comma's frame timeout would never
  /// fire. The Python server waits 5 s, longer than the client's own timeout.
  func drain(_ timeout: TimeInterval) {
    let scratch = UnsafeMutableRawPointer.allocate(byteCount: USBTransport.readChunk, alignment: 64)
    defer { scratch.deallocate() }
    var last = ProcessInfo.processInfo.systemUptime
    while ProcessInfo.processInfo.systemUptime - last < timeout {
      guard let n = try? pipes.read(into: scratch, count: USBTransport.readChunk, timeout: timeout) else { return }
      if n > 0 {
        last = ProcessInfo.processInfo.systemUptime
      }
    }
  }

  // MARK: sending

  func sendParts(_ type: Wire.Msg, seq: UInt32, parts: UnsafeBufferPointer<UnsafeRawBufferPointer>, flags: Wire.Flag) throws {
    sendLock.lock()
    defer { sendLock.unlock() }
    var flags = flags
    var length = 0
    for part in parts { length += part.count }
    let padded = (Wire.headerSize + length) % Wire.packetMultiple == 0
    if padded {
      flags.insert(.padded)
    }
    let total = Wire.headerSize + length + (padded ? 1 : 0)
    if total > txCapacity {
      tx.deallocate()
      txCapacity = max(total, txCapacity * 2)
      tx = UnsafeMutableRawPointer.allocate(byteCount: txCapacity, alignment: 64)
    }
    Wire.packHeader(Wire.Header(msgType: type.rawValue, seq: seq, flags: flags.rawValue, length: UInt32(length)), into: tx)
    var offset = Wire.headerSize
    for part in parts where part.count > 0 {
      (tx + offset).copyMemory(from: part.baseAddress!, byteCount: part.count)
      offset += part.count
    }
    if padded {
      tx.storeBytes(of: UInt8(0), toByteOffset: offset, as: UInt8.self)
    }
    var sent = 0
    while sent < total {
      let n = try pipes.write(from: tx + sent, count: total - sent, timeout: USBTransport.writeTimeout)
      if n <= 0 {
        throw LinkError.closed("peer went away during send")
      }
      sent += n
    }
  }

  // MARK: lifecycle

  func shutdown() {
    pipes.abort()
  }

  func close() {
    pipes.close()
  }
}
