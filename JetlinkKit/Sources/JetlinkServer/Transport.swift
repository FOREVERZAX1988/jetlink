import Foundation

#if canImport(Darwin)
  import Darwin
#elseif canImport(Glibc)
  import Glibc
#endif

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
  private let reader = FrameReader(capacity: 1 << 20)
  private let sendLock = NSLock()
  private let stateLock = NSLock()
  private var closed = false
  /// The outgoing header, and the pad byte after it, packed in place for
  /// every send; `vectors` is reused too, so a frame's reply allocates nothing.
  private let tx: UnsafeMutableRawPointer
  private var vectors: [iovec] = []
  static let maxParts = 6

  init(fd: Int32, peer: String) {
    self.fd = fd
    self.peer = peer
    self.tx = UnsafeMutableRawPointer.allocate(byteCount: Wire.headerSize + 1, alignment: 8)
    self.tx.initializeMemory(as: UInt8.self, repeating: 0, count: Wire.headerSize + 1)
    self.vectors.reserveCapacity(TCPTransport.maxParts + 2)
    TCPTransport.tune(fd)
  }

  deinit {
    close()
    tx.deallocate()
  }

  /// A client's end: what the comma opens, and what a phone dialing the
  /// comma opens. With a `timeout`, a connect that takes longer fails
  /// rather than sit in SYN retries for a minute.
  public static func connect(host: String, port: UInt16, timeout: TimeInterval? = nil) throws -> TCPTransport {
    let fd = socket(AF_INET, Sys.stream, 0)
    guard fd >= 0 else { throw LinkError.closed("socket: \(String(cString: strerror(errno)))") }
    var address = sockaddr_in()
    #if canImport(Darwin)
      address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
    #endif
    address.sin_family = sa_family_t(AF_INET)
    address.sin_port = port.bigEndian
    address.sin_addr.s_addr = inet_addr(host)
    let flags = fcntl(fd, F_GETFL)
    if timeout != nil {
      _ = fcntl(fd, F_SETFL, flags | O_NONBLOCK)
    }
    var connected = withUnsafePointer(to: &address) {
      $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { Sys.connect(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) }
    }
    if connected != 0, let timeout, errno == EINPROGRESS {
      var poller = pollfd(fd: fd, events: Int16(POLLOUT), revents: 0)
      let ready = poll(&poller, 1, Int32(max(1, timeout * 1000)))
      if ready == 0 {
        Sys.close(fd)
        throw LinkError.timedOut("could not connect to \(host):\(port) in \(timeout) s")
      }
      var error: Int32 = 0
      var length = socklen_t(MemoryLayout<Int32>.size)
      if ready > 0 && getsockopt(fd, SOL_SOCKET, SO_ERROR, &error, &length) == 0 && error == 0 {
        connected = 0
      } else {
        errno = error != 0 ? error : errno
      }
    }
    guard connected == 0 else {
      let reason = String(cString: strerror(errno))
      Sys.close(fd)
      throw LinkError.closed("could not connect to \(host):\(port): \(reason)")
    }
    if timeout != nil {
      _ = fcntl(fd, F_SETFL, flags)
    }
    return TCPTransport(fd: fd, peer: "\(host):\(port)")
  }

  /// A receive timeout, so a test waiting on a reply that never comes fails
  /// instead of hanging. The server's side never sets one.
  public func setReceiveTimeout(_ seconds: TimeInterval) {
    var timeout = timeval(tv_sec: .init(seconds), tv_usec: .init((seconds - Double(Int(seconds))) * 1_000_000))
    _ = setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))
  }

  // MARK: receiving

  public func recv() throws -> Message {
    try reader.recv(pad: { Wire.Flag(rawValue: $0.flags).contains(.padded) ? 1 : 0 }) { into, missing, _ in
      while true {
        let n = Sys.read(fd, into, missing)
        if n > 0 { return n }
        if n == 0 { throw LinkError.closed("peer closed the connection") }
        if errno == EINTR { continue }
        if errno == EAGAIN || errno == EWOULDBLOCK {
          throw LinkError.timedOut("only part of a message arrived in time")
        }
        throw LinkError.closed("recv failed: \(String(cString: strerror(errno)))")
      }
    }
  }

  // MARK: sending

  /// One message, `parts` concatenated as its payload, in one vectored write:
  /// header, parts and the pad byte together, so the kernel never sees the
  /// header as a segment of its own. `MessageLink` has the other forms.
  public func sendParts(_ type: Wire.Msg, seq: UInt32, parts: UnsafeBufferPointer<UnsafeRawBufferPointer>, flags: Wire.Flag) throws {
    sendLock.lock()
    defer { sendLock.unlock() }
    var flags = flags
    var length = 0
    for part in parts { length += part.count }
    let padded = Wire.needsPad(length)
    if padded {
      flags.insert(.padded)
    }
    Wire.packHeader(Wire.Header(msgType: type.rawValue, seq: seq, flags: flags.rawValue, length: UInt32(length)), into: tx)
    vectors.removeAll(keepingCapacity: true)
    vectors.append(iovec(iov_base: tx, iov_len: Wire.headerSize))
    for part in parts where part.count > 0 {
      vectors.append(iovec(iov_base: UnsafeMutableRawPointer(mutating: part.baseAddress), iov_len: part.count))
    }
    if padded {
      vectors.append(iovec(iov_base: tx + Wire.headerSize, iov_len: 1))
    }
    try writeAll(&vectors)
  }

  private func writeAll(_ vectors: inout [iovec]) throws {
    var index = 0
    while index < vectors.count {
      let n = vectors.withUnsafeBufferPointer { buffer in
        Sys.writev(fd, buffer.baseAddress! + index, min(vectors.count - index, Sys.iovMax))
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
      _ = Sys.shutdown(fd)
    }
  }

  public func close() {
    stateLock.lock()
    defer { stateLock.unlock() }
    if !closed {
      closed = true
      _ = Sys.shutdown(fd)
      _ = Sys.close(fd)
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
    set(Int32(IPPROTO_TCP), TCP_NODELAY, 1)
    set(SOL_SOCKET, SO_SNDBUF, 4 << 20)
    set(SOL_SOCKET, SO_RCVBUF, 4 << 20)
    #if canImport(Darwin)
      // Linux has no such option; Sys.writev sends with MSG_NOSIGNAL there.
      set(SOL_SOCKET, SO_NOSIGPIPE, 1)
    #endif
    set(SOL_SOCKET, SO_KEEPALIVE, 1)
    set(Int32(IPPROTO_TCP), Sys.keepIdle, 5)
    set(Int32(IPPROTO_TCP), TCP_KEEPINTVL, 2)
    set(Int32(IPPROTO_TCP), TCP_KEEPCNT, 3)
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
    let fd = socket(AF_INET, Sys.stream, 0)
    guard fd >= 0 else { throw LinkError.closed("socket: \(String(cString: strerror(errno)))") }
    var yes: Int32 = 1
    _ = setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, socklen_t(MemoryLayout<Int32>.size))
    #if canImport(Darwin)
      _ = setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &yes, socklen_t(MemoryLayout<Int32>.size))
    #endif
    var address = sockaddr_in()
    #if canImport(Darwin)
      address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
    #endif
    address.sin_family = sa_family_t(AF_INET)
    address.sin_port = port.bigEndian
    address.sin_addr.s_addr = inet_addr(host)
    let bound = withUnsafePointer(to: &address) {
      $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) }
    }
    guard bound == 0, listen(fd, 4) == 0 else {
      let reason = String(cString: strerror(errno))
      Sys.close(fd)
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
      if ready < 0 && errno != EINTR { return nil }
      if ready <= 0 { continue }
      // iOS reclaims a suspended app's listening sockets; this one is gone.
      if poller.revents & Int16(POLLERR | POLLHUP | POLLNVAL) != 0 { return nil }
      var address = sockaddr_in()
      var length = socklen_t(MemoryLayout<sockaddr_in>.size)
      let client = withUnsafeMutablePointer(to: &address) {
        $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { Sys.accept(fd, $0, &length) }
      }
      if client < 0 {
        if errno == EINTR || errno == ECONNABORTED || errno == EAGAIN { continue }
        return nil
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
      Sys.close(fd)
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

/// The socket calls whose names the transports' own methods shadow, and the
/// few constants that differ between Darwin and glibc. Linux is where the
/// conformance suite runs the portable modules (docs/conformance.md).
enum Sys {
  #if canImport(Darwin)
    static let stream = SOCK_STREAM
    static let keepIdle = TCP_KEEPALIVE
    static let iovMax = Int(IOV_MAX)
  #else
    static let stream = Int32(SOCK_STREAM.rawValue)
    static let keepIdle = TCP_KEEPIDLE
    static let iovMax = 1024
  #endif

  @discardableResult
  static func close(_ fd: Int32) -> Int32 {
    #if canImport(Darwin)
      Darwin.close(fd)
    #else
      Glibc.close(fd)
    #endif
  }

  @discardableResult
  static func shutdown(_ fd: Int32) -> Int32 {
    #if canImport(Darwin)
      Darwin.shutdown(fd, SHUT_RDWR)
    #else
      Glibc.shutdown(fd, Int32(SHUT_RDWR))
    #endif
  }

  static func read(_ fd: Int32, _ buffer: UnsafeMutableRawPointer, _ count: Int) -> Int {
    #if canImport(Darwin)
      Darwin.read(fd, buffer, count)
    #else
      Glibc.read(fd, buffer, count)
    #endif
  }

  /// A vectored write that is an error, never SIGPIPE, on a closed socket.
  static func writev(_ fd: Int32, _ vectors: UnsafePointer<iovec>, _ count: Int) -> Int {
    #if canImport(Darwin)
      Darwin.writev(fd, vectors, Int32(count))
    #else
      var message = msghdr()
      message.msg_iov = UnsafeMutablePointer(mutating: vectors)
      message.msg_iovlen = count
      return Glibc.sendmsg(fd, &message, Int32(MSG_NOSIGNAL))
    #endif
  }

  static func connect(_ fd: Int32, _ address: UnsafePointer<sockaddr>, _ length: socklen_t) -> Int32 {
    #if canImport(Darwin)
      Darwin.connect(fd, address, length)
    #else
      Glibc.connect(fd, address, length)
    #endif
  }

  static func accept(_ fd: Int32, _ address: UnsafeMutablePointer<sockaddr>, _ length: UnsafeMutablePointer<socklen_t>) -> Int32 {
    #if canImport(Darwin)
      Darwin.accept(fd, address, length)
    #else
      Glibc.accept(fd, address, length)
    #endif
  }
}
