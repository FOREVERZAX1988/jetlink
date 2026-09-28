import Foundation
import JetlinkKit

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
  static let packetSize = Pinned.usbMaxPacket
  static let readChunk = Pinned.usbReadChunk
  static let readSlack = packetSize
  static let writeTimeout: TimeInterval = 2

  let peer: String
  /// The USB generation the bus negotiated, as the host read it.
  let medium: LinkMedium?
  var connectsOnOpen: Bool { false }
  /// After a protocol error nothing resynchronises the stream, so `close`
  /// drains what the comma is still sending, unless the link was interrupted.
  static let drainTimeout: TimeInterval = 5.0
  private var interrupted = false
  private let pipes: any BulkPipes
  private let reader = FrameReader(capacity: 2 << 20, slack: USBTransport.readSlack)
  var desynced: Bool { reader.desynced }
  private let sendLock = NSLock()
  private var tx: UnsafeMutableRawPointer
  private var txCapacity: Int

  init(pipes: any BulkPipes, peer: String = "usb", medium: LinkMedium? = .usb) {
    self.pipes = pipes
    self.peer = peer
    self.medium = medium
    txCapacity = 1 << 20
    tx = UnsafeMutableRawPointer.allocate(byteCount: txCapacity, alignment: 64)
  }

  deinit {
    pipes.close()
    tx.deallocate()
  }

  // MARK: receiving

  func recv() throws -> Message {
    try reader.recv(pad: { USBTransport.gadgetPad(Wire.headerSize + Int($0.length)) }) { into, missing, room in
      let size = USBTransport.readSize(missing: missing, room: room)
      if size == 0 {
        // Every read would return nothing and the loop would spin while the
        // comma blocks. Say so rather than hang.
        throw LinkError.closed("no room to read the rest of a message (\(missing) bytes to come)")
      }
      return try pipes.read(into: into, count: size, timeout: 0)
    }
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
    let padded = Wire.needsPad(length)
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
    interrupted = true
    pipes.abort()
  }

  /// Closes the pipes. After a desync the comma is still mid-message; it is
  /// let finish and time out first, rather than reopened under.
  func close() {
    if desynced && !interrupted {
      drain(USBTransport.drainTimeout)
    }
    pipes.close()
  }
}
