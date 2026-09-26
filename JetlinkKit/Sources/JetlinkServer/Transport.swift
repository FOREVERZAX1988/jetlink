import Darwin
import Foundation

/// The link is unusable. The comma treats this as "fall back to the small model".
public enum LinkError: Error, CustomStringConvertible {
  case closed(String)
  case timedOut(String)
  case desynced(String)

  public var description: String {
    switch self {
    case .closed(let detail), .timedOut(let detail), .desynced(let detail): return detail
    }
  }
}

/// One received message. The payload is a view into the transport's receive
/// buffer, valid only until the next `recv()`: copy anything you keep.
public struct Message {
  public let msgType: UInt16
  public let seq: UInt32
  public let flags: UInt32
  public let payload: UnsafeRawBufferPointer
}

/// Framing over a connected TCP socket, as `StreamTransport` and `TcpTransport`
/// do it in Python.
///
/// Reads never go past the end of the current message, so every message
/// lands at the start of the receive buffer and the float32 arrays inside an
/// INFER stay aligned. The steady state neither allocates nor copies.
public final class TCPTransport: @unchecked Sendable {
  public let peer: String
  private let fd: Int32
  private var rx: UnsafeMutableRawPointer
  private var capacity: Int
  private var start = 0
  private var end = 0
  private var desynced = false
  private let sendLock = NSLock()
  private let stateLock = NSLock()
  private var closed = false

  init(fd: Int32, peer: String) {
    self.fd = fd
    self.peer = peer
    self.capacity = 1 << 20
    self.rx = UnsafeMutableRawPointer.allocate(byteCount: capacity, alignment: 64)
    TCPTransport.tune(fd)
  }

  deinit {
    close()
    rx.deallocate()
  }

  // MARK: receiving

  public func recv() throws -> Message {
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
      // Nothing resynchronises a byte stream mid-message.
      desynced = true
      throw LinkError.desynced("protocol error, link unusable: \(error)")
    }
    let pad = Wire.Flag(rawValue: header.flags).contains(.padded) ? 1 : 0
    let total = Wire.headerSize + Int(header.length) + pad
    try fill(total)
    let payload = UnsafeRawBufferPointer(start: rx + start + Wire.headerSize, count: Int(header.length))
    start += total
    if start == end {
      start = 0
      end = 0
    }
    return Message(msgType: header.msgType, seq: header.seq, flags: header.flags, payload: payload)
  }

  /// Read until `need` bytes of the current message are buffered.
  private func fill(_ need: Int) throws {
    reserve(need)
    while end - start < need {
      let want = need - (end - start)
      let n = Darwin.read(fd, rx + end, want)
      if n > 0 {
        end += n
      } else if n == 0 {
        throw LinkError.closed("peer closed the connection")
      } else if errno == EINTR {
        continue
      } else if errno == EAGAIN || errno == EWOULDBLOCK {
        throw LinkError.timedOut("only \(end - start) of \(need) bytes arrived in time")
      } else {
        throw LinkError.closed("recv failed: \(String(cString: strerror(errno)))")
      }
    }
  }

  private func reserve(_ need: Int) {
    if start > 0 && start + need > capacity {
      memmove(rx, rx + start, end - start)
      end -= start
      start = 0
    }
    if need > capacity {
      let grown = UnsafeMutableRawPointer.allocate(byteCount: max(need, capacity * 2), alignment: 64)
      grown.copyMemory(from: rx + start, byteCount: end - start)
      rx.deallocate()
      rx = grown
      capacity = max(need, capacity * 2)
      end -= start
      start = 0
    }
  }

  // MARK: sending

  /// One message, `parts` concatenated as its payload, in one vectored write.
  public func send(_ type: Wire.Msg, seq: UInt32, parts: [UnsafeRawBufferPointer] = [], flags: Wire.Flag = []) throws {
    sendLock.lock()
    defer { sendLock.unlock() }
    var flags = flags
    let length = parts.reduce(0) { $0 + $1.count }
    let padded = (Wire.headerSize + length) % Wire.packetMultiple == 0
    if padded {
      flags.insert(.padded)
    }
    var header = [UInt8](repeating: 0, count: Wire.headerSize)
    var pad: UInt8 = 0
    try header.withUnsafeMutableBytes { headerBytes in
      Wire.packHeader(Wire.Header(msgType: type.rawValue, seq: seq, flags: flags.rawValue, length: UInt32(length)), into: headerBytes.baseAddress!)
      try withUnsafeMutablePointer(to: &pad) { padPointer in
        var vectors: [iovec] = [iovec(iov_base: headerBytes.baseAddress, iov_len: Wire.headerSize)]
        for part in parts where part.count > 0 {
          vectors.append(iovec(iov_base: UnsafeMutableRawPointer(mutating: part.baseAddress), iov_len: part.count))
        }
        if padded {
          vectors.append(iovec(iov_base: UnsafeMutableRawPointer(padPointer), iov_len: 1))
        }
        try writeAll(&vectors)
      }
    }
  }

  public func send(_ type: Wire.Msg, seq: UInt32, data: [Data], flags: Wire.Flag = []) throws {
    try Self.withBuffers(data) { try send(type, seq: seq, parts: $0, flags: flags) }
  }

  public func sendJSON(_ type: Wire.Msg, seq: UInt32, _ object: [String: Any], flags: Wire.Flag = []) throws {
    try send(type, seq: seq, data: [JSONLine.encode(object)], flags: flags)
  }

  private static func withBuffers<R>(_ data: [Data], _ body: ([UnsafeRawBufferPointer]) throws -> R) throws -> R {
    var buffers: [UnsafeRawBufferPointer] = []
    func recurse(_ index: Int) throws -> R {
      if index == data.count {
        return try body(buffers)
      }
      return try data[index].withUnsafeBytes { bytes in
        buffers.append(bytes)
        return try recurse(index + 1)
      }
    }
    return try recurse(0)
  }

  private func writeAll(_ vectors: inout [iovec]) throws {
    var index = 0
    while index < vectors.count {
      let n = vectors.withUnsafeBufferPointer { buffer in
        Darwin.writev(fd, buffer.baseAddress! + index, Int32(min(vectors.count - index, Int(IOV_MAX))))
      }
      if n < 0 {
        if errno == EINTR { continue }
        if errno == EAGAIN || errno == EWOULDBLOCK {
          throw LinkError.timedOut("send timed out; link abandoned")
        }
        throw LinkError.closed("send failed: \(String(cString: strerror(errno)))")
      }
      if n == 0 {
        throw LinkError.closed("peer went away during send")
      }
      var remaining = n
      while remaining > 0 && index < vectors.count {
        if remaining >= vectors[index].iov_len {
          remaining -= vectors[index].iov_len
          index += 1
        } else {
          vectors[index].iov_base = vectors[index].iov_base.map { $0 + remaining }
          vectors[index].iov_len -= remaining
          remaining = 0
        }
      }
    }
  }

  // MARK: lifecycle

  /// Wakes a `recv()` blocked on another thread; it then throws `.closed`.
  public func shutdown() {
    stateLock.lock()
    defer { stateLock.unlock() }
    if !closed {
      _ = Darwin.shutdown(fd, SHUT_RDWR)
    }
  }

  public func close() {
    stateLock.lock()
    defer { stateLock.unlock() }
    if !closed {
      closed = true
      _ = Darwin.shutdown(fd, SHUT_RDWR)
      _ = Darwin.close(fd)
    }
  }

  /// NODELAY is the one that matters: without it the header and the body can be
  /// split across an RTT. The keepalive is short because a pulled cable
  /// otherwise leaves the session blocked in recv for hours, and the comma's
  /// reconnect waiting behind it.
  private static func tune(_ fd: Int32) {
    func set(_ level: Int32, _ option: Int32, _ value: Int32) {
      var value = value
      _ = setsockopt(fd, level, option, &value, socklen_t(MemoryLayout<Int32>.size))
    }
    set(IPPROTO_TCP, TCP_NODELAY, 1)
    set(SOL_SOCKET, SO_SNDBUF, 4 << 20)
    set(SOL_SOCKET, SO_RCVBUF, 4 << 20)
    set(SOL_SOCKET, SO_NOSIGPIPE, 1)
    set(SOL_SOCKET, SO_KEEPALIVE, 1)
    set(IPPROTO_TCP, TCP_KEEPALIVE, 5)
    set(IPPROTO_TCP, TCP_KEEPINTVL, 2)
    set(IPPROTO_TCP, TCP_KEEPCNT, 3)
    var timeout = timeval(tv_sec: 10, tv_usec: 0)
    _ = setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))
  }
}

/// A listening TCP socket. `accept` polls so `close` can stop it from another thread.
public final class TCPListener: @unchecked Sendable {
  public let port: UInt16
  private let fd: Int32
  private let lock = NSLock()
  private var closed = false

  public init(host: String = "0.0.0.0", port: UInt16 = Wire.defaultPort) throws {
    let fd = socket(AF_INET, SOCK_STREAM, 0)
    guard fd >= 0 else { throw LinkError.closed("socket: \(String(cString: strerror(errno)))") }
    var yes: Int32 = 1
    _ = setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, socklen_t(MemoryLayout<Int32>.size))
    _ = setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &yes, socklen_t(MemoryLayout<Int32>.size))
    var address = sockaddr_in()
    address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
    address.sin_family = sa_family_t(AF_INET)
    address.sin_port = port.bigEndian
    address.sin_addr.s_addr = inet_addr(host)
    let bound = withUnsafePointer(to: &address) {
      $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) }
    }
    guard bound == 0, listen(fd, 4) == 0 else {
      let reason = String(cString: strerror(errno))
      Darwin.close(fd)
      throw LinkError.closed("could not listen on \(host):\(port): \(reason)")
    }
    var actual = sockaddr_in()
    var length = socklen_t(MemoryLayout<sockaddr_in>.size)
    _ = withUnsafeMutablePointer(to: &actual) {
      $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { getsockname(fd, $0, &length) }
    }
    self.fd = fd
    self.port = UInt16(bigEndian: actual.sin_port)
  }

  deinit {
    close()
  }

  /// The next client, or nil once the listener is closed. Blocks.
  public func accept() -> TCPTransport? {
    while true {
      lock.lock()
      let isClosed = closed
      lock.unlock()
      if isClosed { return nil }
      var poller = pollfd(fd: fd, events: Int16(POLLIN), revents: 0)
      let ready = poll(&poller, 1, 250)
      if ready <= 0 { continue }
      var address = sockaddr_in()
      var length = socklen_t(MemoryLayout<sockaddr_in>.size)
      let client = withUnsafeMutablePointer(to: &address) {
        $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { Darwin.accept(fd, $0, &length) }
      }
      if client < 0 {
        if errno == EINTR || errno == ECONNABORTED { continue }
        lock.lock()
        let isClosed = closed
        lock.unlock()
        if isClosed { return nil }
        continue
      }
      var text = [CChar](repeating: 0, count: Int(INET_ADDRSTRLEN))
      inet_ntop(AF_INET, &address.sin_addr, &text, socklen_t(INET_ADDRSTRLEN))
      let peer = "\(String(decoding: text.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)):\(UInt16(bigEndian: address.sin_port))"
      return TCPTransport(fd: client, peer: peer)
    }
  }

  public func close() {
    lock.lock()
    defer { lock.unlock() }
    if !closed {
      closed = true
      Darwin.close(fd)
    }
  }
}

/// JSON the way the Python server writes it, for payloads and control lines.
enum JSONLine {
  static func encode(_ object: [String: Any]) -> Data {
    (try? JSONSerialization.data(withJSONObject: object, options: [.withoutEscapingSlashes])) ?? Data("{}".utf8)
  }

  static func decode(_ bytes: UnsafeRawBufferPointer) -> [String: Any]? {
    guard bytes.count > 0 else { return [:] }
    let data = Data(bytes)
    return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
  }
}
