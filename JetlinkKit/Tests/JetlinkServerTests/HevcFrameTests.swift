import CHevc
import Foundation
import Testing

@testable import JetlinkServer

/// The HEVC frame path: the two cameras' access units back to the six
/// planes the model takes.
///
/// The decoder behind it is libavcodec, opened at run time, so a host
/// without it has no decoder and the tests that need one are skipped rather
/// than failed: the server says so in its hello, and a link never negotiates
/// a codec this side cannot take.
@Suite("HEVC frames")
struct HevcFrameTests {
  /// A frame of `shape`, made if this host can decode.
  private func frame(_ shape: [Int]) -> HevcFrame? {
    guard HevcFrame.available else { return nil }
    return HevcFrame(shape: shape)
  }

  @Test("the codec is the one the Python names")
  func theCodecName() {
    #expect(HevcFrame.codec == "hevc")
  }

  @Test("a frame that is not a warped frame's shape is refused")
  func aWrongShapeIsRefused() {
    // not (2, 6, H, W): no camera, no six planes, a negative side
    #expect(HevcFrame(shape: [6, 2, 128, 256]) == nil)
    #expect(HevcFrame(shape: [2, 6, 128]) == nil)
    #expect(HevcFrame(shape: [2, 6, 0, 256]) == nil)
    #expect(HevcFrame(shape: [2, 6, -1, 256]) == nil)
  }

  @Test("a frame of the right shape is made, and the NV12 is twice a plane")
  func theShapeOfTheFrame() throws {
    guard let frame = frame([2, 6, 128, 256]) else { return }
    #expect(frame.planeHeight == 128)
    #expect(frame.planeWidth == 256)
    // the encoder's NV12 is the luma twice a plane wide and tall
    #expect(frame.width == 512)
    #expect(frame.height == 256)
  }

  @Test("a payload that does not add up is refused")
  func aBadPayloadIsRefused() throws {
    guard let frame = frame([2, 6, 128, 256]) else { return }
    // the two lengths, then nothing: the lengths promise more than is there
    var payload = [UInt8](repeating: 0, count: 8)
    payload[0] = 100
    payload[4] = 100
    #expect(!frame.unpack(UnsafeRawBufferPointer(bytes: payload, count: payload.count)))
    // a length past the message
    payload = [UInt8](repeating: 0, count: 16)
    payload[0] = 0xFF
    payload[4] = 0xFF
    #expect(!frame.unpack(UnsafeRawBufferPointer(bytes: payload, count: payload.count)))
    // too short to hold the two lengths
    #expect(!frame.unpack(UnsafeRawBufferPointer(bytes: [0, 0, 0], count: 3)))
  }

  @Test("the bitstream half reads the access unit's NAL units")
  func theNalUnits() {
    // a VPS, an SPS and an IDR, each behind a three-byte start code
    let au: [UInt8] = [
      0x00, 0x00, 0x01, 0x40, 0x01, 0x0C, 0x01,
      0x00, 0x00, 0x01, 0x42, 0x01, 0x01,
      0x00, 0x00, 0x01, 0x26, 0x01, 0xAF, 0x08,
    ]
    var nals = [jl_nal](repeating: jl_nal(), count: 8)
    let count = au.withUnsafeBytes { jl_hevc_nals($0.baseAddress!, $0.count, &nals, 8) }
    #expect(count == 3)
    #expect(nals[0].type == 32)  // the VPS
    #expect(nals[1].type == 33)  // the SPS
    #expect(nals[2].type == 19)  // the IDR
    #expect(au.withUnsafeBytes { jl_hevc_is_keyframe($0.baseAddress!, $0.count) } == 1)
    // a P frame alone is not a keyframe
    let p: [UInt8] = [0x00, 0x00, 0x01, 0x02, 0x01, 0xD0]
    #expect(p.withUnsafeBytes { jl_hevc_is_keyframe($0.baseAddress!, $0.count) } == 0)
  }

  @Test("the NV12 unpacks into the six planes, each in its own place")
  func theUnpack() {
    // a frame of 8 x 4 holds one plane of 4 x 2, so the six planes are 8
    // bytes each
    let width = 8, height = 4
    let frameBytes = jl_hevc_frame_bytes(Int32(width), Int32(height))
    #expect(frameBytes == 48)
    var nv12 = [UInt8](repeating: 0, count: frameBytes)
    for y in 0..<height {
      for x in 0..<width {
        nv12[y * width + x] = UInt8(y * 10 + x)
      }
    }
    let luma = width * height
    for y in 0..<(height / 2) {
      for x in 0..<(width / 2) {
        nv12[luma + (y * (width / 2) + x) * 2] = UInt8(100 + y)
        nv12[luma + (y * (width / 2) + x) * 2 + 1] = UInt8(200 + y)
      }
    }
    var planes = [UInt8](repeating: 0, count: 6 * 8)
    nv12.withUnsafeBytes { source in
      planes.withUnsafeMutableBytes { out in
        jl_hevc_unpack_nv12(source.baseAddress!, Int32(width), Int32(height), out.baseAddress!)
      }
    }
    // the luma's four phases: even,even then even,odd then odd,even then odd,odd
    #expect(planes[0] == 0 && planes[1] == 2)
    #expect(planes[8] == 1 && planes[9] == 3)
    #expect(planes[16] == 10 && planes[17] == 12)
    #expect(planes[24] == 11 && planes[25] == 13)
    // and the chroma's two planes
    #expect(planes[32] == 100)
    #expect(planes[40] == 200)
  }

  @Test("the server offers the codec only where it can decode")
  func theCodecIsOfferedOnlyWhereItCanBeTaken() {
    let codecs = Session.frameCodecs()
    #expect(codecs.contains(LosslessFrame.codec))
    // the packed one is always there; the HEVC one only with a decoder
    #expect(codecs.contains(HevcFrame.codec) == HevcFrame.available)
  }
}
