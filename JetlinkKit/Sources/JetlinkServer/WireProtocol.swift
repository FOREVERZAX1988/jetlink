import Foundation
import JetlinkKit

/// The wire protocol, byte for byte what `jetlink/protocol.py` defines.
///
/// Every message is a 32-byte header and an opaque payload. INFER is a fixed
/// struct plus two raw arrays sized at handshake, not json: one vectored write
/// out, one read into a preallocated buffer back. The numbers come from
/// `Pinned`, which make_pins.py writes from the Python; the enums below are
/// checked against it in ConformanceTests.
public enum Wire {
  public static let magic = Pinned.magic  // b'JLNK'
  /// The one version this server speaks: the comma package and the server
  /// are updated together. 3 keeps the hidden state here, so INFER_REQ has no
  /// prev_feat and INFER_RESP no hidden_state slice.
  public static let version = Pinned.protocolVersion
  /// The header version of the envelope: the messages any two versions must
  /// still exchange. A server of version 2 latches any other as a desync, so
  /// a newer comma's hello would hang there instead of failing.
  public static let envelopeVersion = Pinned.envelopeVersion
  /// The hello, where a mismatch is found and named; the shutdown, which the
  /// comma sends with no hello when its battery runs low; and the ERROR either
  /// may get back. Their bodies are JSON and read the same in every version.
  public static let envelope = Set(Pinned.envelopeMessages)
  public static let headerSize = Pinned.headerSize
  /// A bulk transfer ends on a short packet, so a message that is an exact
  /// multiple of the packet size gets a pad byte and Flag.padded. TCP keeps
  /// the rule so one client speaks to every transport the same way.
  public static let packetMultiple = Pinned.packetMultiple
  /// The gadget pads every message it sends to a whole burst, so none ends on
  /// a short packet: dwc3 flushed its TX FIFO past one about once in 400
  /// frames. A USB host reads each message to that boundary. Host to device
  /// keeps the one-byte pad instead. `protocol.GADGET_TX_ALIGN`.
  public static let gadgetTxAlign = Pinned.gadgetTxAlign

  /// Does a sent message with a `length` byte payload need the pad byte and
  /// Flag.padded? The rule every sender but the gadget keeps.
  static func needsPad(_ length: Int) -> Bool {
    (headerSize + length) % packetMultiple == 0
  }
  /// Stops a corrupt length field making the receive buffer allocate
  /// gigabytes. A frame is about 460 KB.
  public static let maxMessage = Pinned.maxMessage
  public static let defaultPort = Pinned.defaultPort

  public enum Msg: UInt16, Sendable {
    case helloReq = 1
    case helloResp = 2
    case engineReq = 3
    case engineResp = 4
    case uploadChunk = 5
    case uploadDone = 6
    case progress = 7
    case inferReq = 8
    case inferResp = 9
    case stateReq = 12
    case stateResp = 13
    case error = 14
    case ping = 15
    case pong = 16
    case shutdownReq = 17
    case shutdownResp = 18
  }

  public struct Flag: OptionSet, Sendable {
    public let rawValue: UInt32
    public init(rawValue: UInt32) { self.rawValue = rawValue }
    /// On INFER_REQ: warm-start, clear history before this frame.
    public static let resetQueues = Flag(rawValue: 1 << 0)
    /// On INFER_REQ: append telemetry json to the response.
    public static let wantState = Flag(rawValue: 1 << 1)
    /// On INFER_REQ: keep hidden_state in the response, for a comma logging
    /// the whole output vector.
    public static let wantHidden = Flag(rawValue: 1 << 2)
    /// One pad byte follows the payload; see packetMultiple.
    public static let padded = Flag(rawValue: 1 << 7)
  }

  public enum Status: UInt32, Sendable {
    case ok = 0
    case notReady = 1
    case badShape = 2
    case inferFailed = 3
    /// The model produced NaN or Inf; the comma falls back.
    case notFinite = 4
  }

  /// INFER_REQ: frame_id, flags.
  public static let inferReqSize = Pinned.inferReqSize
  /// INFER_RESP: frame_id, status, gpu_us, queue_us, total_us.
  public static let inferRespSize = Pinned.inferRespSize

  public struct Header: Equatable, Sendable {
    public var msgType: UInt16
    public var seq: UInt32
    public var flags: UInt32
    public var length: UInt32
    public var reserved: UInt64 = 0
    /// What the message is sent with, or was received with.
    public var version: UInt16

    public init(msgType: UInt16, seq: UInt32, flags: UInt32, length: UInt32, reserved: UInt64 = 0, version: UInt16? = nil) {
      self.msgType = msgType
      self.seq = seq
      self.flags = flags
      self.length = length
      self.reserved = reserved
      self.version = version ?? Wire.headerVersion(msgType)
    }
  }

  /// The version a message of this type is sent with.
  public static func headerVersion(_ msgType: UInt16) -> UInt16 {
    envelope.contains(msgType) ? envelopeVersion : version
  }

  /// Did a peer on this protocol send this message? Only envelope messages
  /// may carry the envelope's version.
  public static func sameProtocol(version received: UInt16, msgType: UInt16) -> Bool {
    received == version || (received == envelopeVersion && envelope.contains(msgType))
  }

  /// What this server says of a mismatch, naming the side to update.
  public static func updateHint(comma: Int) -> String {
    let side = comma < Int(version) ? "update the comma's jetlink package" : "update jetlink on this server"
    return "this server runs jetlink protocol \(version) and the comma \(comma): \(side)"
  }

  public enum ProtocolError: Error, CustomStringConvertible {
    case badMagic(UInt32)
    case badVersion(UInt16)
    case tooLong(UInt32)

    public var description: String {
      switch self {
      case .badMagic(let magic):
        return "bad magic 0x\(String(magic, radix: 16)) (link desynced or not a jetlink peer)"
      case .badVersion(let version):
        return "peer speaks protocol v\(version), we speak v\(Wire.version)"
      case .tooLong(let length):
        return "message claims \(length) bytes, over the \(Wire.maxMessage) cap"
      }
    }
  }

  /// '<IHHIIIQ4x': magic, version, msg_type, seq, flags, length, reserved, 4 pad.
  public static func packHeader(_ header: Header, into out: UnsafeMutableRawPointer) {
    out.storeBytes(of: magic.littleEndian, toByteOffset: 0, as: UInt32.self)
    out.storeBytes(of: header.version.littleEndian, toByteOffset: 4, as: UInt16.self)
    out.storeBytes(of: header.msgType.littleEndian, toByteOffset: 6, as: UInt16.self)
    out.storeBytes(of: header.seq.littleEndian, toByteOffset: 8, as: UInt32.self)
    out.storeBytes(of: header.flags.littleEndian, toByteOffset: 12, as: UInt32.self)
    out.storeBytes(of: header.length.littleEndian, toByteOffset: 16, as: UInt32.self)
    out.storeBytes(of: header.reserved.littleEndian, toByteOffset: 20, as: UInt64.self)
    out.storeBytes(of: UInt32(0), toByteOffset: 28, as: UInt32.self)
  }

  /// The header's fields. A version-2 header is taken on any message: the
  /// two frame alike, so the message is read whole, the stream stays in
  /// sync, and the session refuses it by name. Any other version is a desync.
  public static func unpackHeader(_ buffer: UnsafeRawPointer) throws -> Header {
    let magicRead = UInt32(littleEndian: buffer.loadUnaligned(fromByteOffset: 0, as: UInt32.self))
    guard magicRead == magic else { throw ProtocolError.badMagic(magicRead) }
    let versionRead = UInt16(littleEndian: buffer.loadUnaligned(fromByteOffset: 4, as: UInt16.self))
    guard versionRead == version || versionRead == envelopeVersion else { throw ProtocolError.badVersion(versionRead) }
    return Header(
      msgType: UInt16(littleEndian: buffer.loadUnaligned(fromByteOffset: 6, as: UInt16.self)),
      seq: UInt32(littleEndian: buffer.loadUnaligned(fromByteOffset: 8, as: UInt32.self)),
      flags: UInt32(littleEndian: buffer.loadUnaligned(fromByteOffset: 12, as: UInt32.self)),
      length: UInt32(littleEndian: buffer.loadUnaligned(fromByteOffset: 16, as: UInt32.self)),
      reserved: UInt64(littleEndian: buffer.loadUnaligned(fromByteOffset: 20, as: UInt64.self)),
      version: versionRead)
  }

  public static func packInferResp(frameID: UInt32, status: Status, gpuUs: UInt32, queueUs: UInt32, totalUs: UInt32, into out: UnsafeMutableRawPointer) {
    out.storeBytes(of: frameID.littleEndian, toByteOffset: 0, as: UInt32.self)
    out.storeBytes(of: status.rawValue.littleEndian, toByteOffset: 4, as: UInt32.self)
    out.storeBytes(of: gpuUs.littleEndian, toByteOffset: 8, as: UInt32.self)
    out.storeBytes(of: queueUs.littleEndian, toByteOffset: 12, as: UInt32.self)
    out.storeBytes(of: totalUs.littleEndian, toByteOffset: 16, as: UInt32.self)
  }

  public static func inferResp(frameID: UInt32 = 0, status: Status, gpuUs: UInt32 = 0, queueUs: UInt32 = 0, totalUs: UInt32 = 0) -> [UInt8] {
    var bytes = [UInt8](repeating: 0, count: inferRespSize)
    bytes.withUnsafeMutableBytes {
      packInferResp(frameID: frameID, status: status, gpuUs: gpuUs, queueUs: queueUs, totalUs: totalUs, into: $0.baseAddress!)
    }
    return bytes
  }
}
