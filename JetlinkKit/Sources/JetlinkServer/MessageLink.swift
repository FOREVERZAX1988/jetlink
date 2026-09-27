import Foundation

/// One framed, ordered message channel to one comma: a TCP connection, or bulk
/// transfers to the comma's USB gadget. The session speaks to either the same
/// way; only how the bytes are framed on the way differs.
protocol MessageLink: AnyObject, Sendable {
  /// Who is on the other end, for the link event and the log.
  var peer: String { get }
  /// The next message. Its payload is a view into the link's receive buffer,
  /// valid only until the next `recv()`.
  func recv() throws -> Message
  /// One message, `parts` concatenated as its payload, from the caller's storage.
  func sendParts(_ type: Wire.Msg, seq: UInt32, parts: UnsafeBufferPointer<UnsafeRawBufferPointer>, flags: Wire.Flag) throws
  /// Wakes a `recv()` blocked on another thread, which then throws.
  func shutdown()
  func close()
}

extension MessageLink {
  func send(_ type: Wire.Msg, seq: UInt32, parts: [UnsafeRawBufferPointer] = [], flags: Wire.Flag = []) throws {
    try parts.withUnsafeBufferPointer { try sendParts(type, seq: seq, parts: $0, flags: flags) }
  }

  func sendJSON(_ type: Wire.Msg, seq: UInt32, _ object: [String: Any], flags: Wire.Flag = []) throws {
    let data = JSONLine.encode(object)
    try data.withUnsafeBytes { bytes in
      try send(type, seq: seq, parts: [bytes], flags: flags)
    }
  }
}

extension TCPTransport: MessageLink {
  func sendParts(_ type: Wire.Msg, seq: UInt32, parts: UnsafeBufferPointer<UnsafeRawBufferPointer>, flags: Wire.Flag) throws {
    try send(type, seq: seq, parts: parts, flags: flags)
  }
}
