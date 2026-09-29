import Foundation

/// The receiving half of a framed byte stream, shared by TCP and USB: the
/// buffer, the header checks and the desync latch. A transport says only
/// what differs: the pad after a message, and how to read the bytes missing.
///
/// Reads never go past what the transport asks for, and each message is
/// consumed whole, so every message lands at the start of the buffer and the
/// float32 arrays inside an INFER stay aligned. The steady state neither
/// allocates nor copies.
final class FrameReader {
  private var rx: UnsafeMutableRawPointer
  private var capacity: Int
  private var start = 0
  private var end = 0
  /// Room kept past a message, for transports whose reads come in whole
  /// packets (USB).
  private let slack: Int
  /// Nothing resynchronises a byte stream mid-message, so one protocol error
  /// is the end of the link.
  private(set) var desynced = false

  init(capacity: Int, slack: Int = 0) {
    self.capacity = capacity
    self.slack = slack
    rx = UnsafeMutableRawPointer.allocate(byteCount: capacity, alignment: 64)
  }

  deinit {
    rx.deallocate()
  }

  /// The next message, its payload a view valid until the next call. `pad`
  /// is how many bytes follow a message's payload; `read` reads into its
  /// pointer toward `missing` more bytes, with `room` bytes free there, and
  /// returns how many arrived.
  func recv(
    pad: (Wire.Header) -> Int, read: (_ into: UnsafeMutableRawPointer, _ missing: Int, _ room: Int) throws -> Int
  ) throws -> Message {
    if desynced {
      throw LinkError.desynced("stream desynced; the link must be reopened")
    }
    try fill(Wire.headerSize, read)
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
    let total = Wire.headerSize + Int(header.length) + pad(header)
    try fill(total, read)
    let payload = UnsafeRawBufferPointer(start: rx + start + Wire.headerSize, count: Int(header.length))
    start += total
    if start == end {
      start = 0
      end = 0
    }
    return Message(msgType: header.msgType, seq: header.seq, flags: header.flags, payload: payload, version: header.version)
  }

  /// Reads until `need` bytes of the current message are buffered.
  private func fill(_ need: Int, _ read: (UnsafeMutableRawPointer, Int, Int) throws -> Int) throws {
    reserve(need + slack)
    while end - start < need {
      end += try read(rx + end, need - (end - start), capacity - end)
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
}
