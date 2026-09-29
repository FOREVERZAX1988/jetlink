import Foundation
import JetlinkKit
import Testing

@testable import JetlinkServer

/// Protocol 3 on the server's side: the hidden state stays here, the frames
/// are smaller, and a header of any other version is a broken stream.
@Suite("Protocol 3", .serialized)
struct ProtocolTests {
  /// A message as the comma's gadget sends it, 16 KB-padded, with the
  /// header version given.
  func gadgetFrame(_ type: Wire.Msg, seq: UInt32, payload: Data = Data(), version: UInt16 = Wire.version) -> [UInt8] {
    var bytes = [UInt8](repeating: 0, count: Wire.headerSize)
    bytes.withUnsafeMutableBytes {
      Wire.packHeader(Wire.Header(msgType: type.rawValue, seq: seq, flags: 0, length: UInt32(payload.count)), into: $0.baseAddress!)
      $0.storeBytes(of: version.littleEndian, toByteOffset: 4, as: UInt16.self)
    }
    bytes += payload
    bytes += [UInt8](repeating: 0, count: USBTransport.gadgetPad(bytes.count))
    return bytes
  }

  /// A 766 MB model's protocol-3 request is 409,600 bytes as the gadget pads
  /// it: 25 of the ring's reads, all posted before it came, none posted while
  /// it was read. A header of another version after it is a broken stream,
  /// as any bad header: no reply, the link latched desynced.
  @Test("Protocol 3 through the host's read ring")
  func throughTheRing() throws {
    let kernel = FakeUsbfs()
    let pipes = UsbfsPipes(device: UsbfsDevice(kernel: kernel), inEndpoint: 0x81, outEndpoint: 0x01, depth: ReadRing.depth)
    let scratch = Scratch(64)
    #expect(try pipes.read(into: scratch.pointer, count: 64, timeout: 0.01) == 0)
    #expect(kernel.pendingCount == ReadRing.depth)
    let transport = USBTransport(pipes: pipes)

    let request = Data(pattern(Wire.inferReqSize + 2 * 6 * 128 * 256 + 12 * 4))
    let frame = gadgetFrame(.inferReq, seq: 1, payload: request)
    #expect(frame.count == 409_600 && frame.count == 25 * ReadRing.slotSize)
    kernel.feed(frame)
    let message = try transport.recv()
    #expect(message.msgType == Wire.Msg.inferReq.rawValue)
    #expect(Data(message.payload) == request)
    #expect(kernel.pendingCount == ReadRing.depth - 25)
    #expect(kernel.discards == 0)

    kernel.feed(gadgetFrame(.helloReq, seq: 2, payload: Data("{}".utf8), version: Wire.version - 1))
    #expect(throws: LinkError.self) { try transport.recv() }
    #expect(transport.desynced)
    #expect(kernel.written.isEmpty)
  }

  /// The queued golden frames carry a prev_feat of their own, as protocol 2
  /// sent it. Taken in through the server's own feedback path, as a hidden
  /// state kept from the frame before, they give protocol 2's outputs: the
  /// staging and the engine are unchanged, only where the hidden state comes
  /// from is.
  @Test("Protocol 2's outputs, from the server's own feedback path")
  func protocol2Outputs() throws {
    let golden = try Golden("tiny_queued")
    let cache = try TemporaryDirectory()
    let server = try Server(
      configuration: Server.Configuration(host: "127.0.0.1", port: 0, cacheRoot: cache.url, preload: false), backend: cpuBackend())
    try server.start()
    defer { server.stop() }
    let client = try TestClient(port: server.port!)
    let ready = try client.ensureEngine(model: golden.model, sha256: golden.sha256)
    client.close()
    let spec = try ModelSpec.from(ready["spec"] as! [String: Any])
    let hidden = try #require(spec.hiddenRange)

    server.host.lock.lock()
    defer { server.host.lock.unlock() }
    let loaded = try #require(server.host.loaded)
    let type = try #require(loaded.engine.outputs[ModelConstants.drivingOutput]?.type)
    var kept = [Float](repeating: 0, count: spec.outputCount)
    var output = [Float](repeating: 0, count: spec.outputCount)
    let frameBytes = golden.frameBytes(spec)
    loaded.staging.reset()
    for i in 0..<(golden.frames.count / frameBytes) {
      try golden.frames.withUnsafeBytes { raw in
        let frame = raw.baseAddress! + i * frameBytes
        let packed = frame + spec.warpedBytes
        // the recorded prev_feat, after the scalars, as the last frame's hidden state
        kept.withUnsafeMutableBytes { k in
          (k.baseAddress! + hidden.lowerBound * 4).copyMemory(from: packed + spec.packedBytes, byteCount: hidden.count * 4)
        }
        kept.withUnsafeBufferPointer { loaded.staging.keep(outputs: $0.baseAddress!) }
        try loaded.staging.stage(warped: frame, packed: packed)
      }
      try loaded.engine.run()
      let out = try #require(loaded.engine.output(ModelConstants.drivingOutput))
      output.withUnsafeMutableBytes { o in
        if type == .float16 {
          Convert.f16ToF32(out, o.baseAddress!, count: spec.outputCount)
        } else {
          o.baseAddress!.copyMemory(from: out, byteCount: spec.outputBytes)
        }
      }
      let got = output.withUnsafeBytes { Data($0) }
      let expected = Data(golden.expected[(i * spec.outputBytes)..<((i + 1) * spec.outputBytes)])
      #if os(Android) || os(Linux)
        // fp16 arithmetic there; see CommaClient.replay
        #expect(Golden.correlation(got, expected) >= 0.999, "frame \(i)")
      #else
        #expect(got == expected, "frame \(i) differs from protocol 2's by up to \(Golden.worstDifference(got, expected))")
      #endif
    }
  }
}
