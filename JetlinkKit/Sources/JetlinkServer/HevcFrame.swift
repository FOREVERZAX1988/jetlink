import CHevc
import Foundation
import JetlinkKit

/// An HEVC frame (`Wire.Flag.hevc`) back to the warped frame, bit for bit
/// as the encoder made it: the six planes the model was built for, in the
/// order the warp put them.
///
/// The two cameras' access units travel in one INFER_REQ, each behind its
/// own u32 length, and each is a camera's own stream: they have their own
/// decoders, and a frame that has not been decoded yet is not published.
///
/// The Python is `jetlink.hevc.from_nv12`, and the C is `CHevc`.
final class HevcFrame: @unchecked Sendable {
  /// The codec's name, as the hello names it.
  static let codec = Pinned.hevcCodec

  /// The width and height of the NV12 the encoder made: the model's own
  /// plane is half of each, and the luma is the four planes interleaved.
  let width: Int
  let height: Int
  /// One plane's shape: (height / 2, width / 2).
  let planeHeight: Int
  let planeWidth: Int

  private var road: OpaquePointer?
  private var wide: OpaquePointer?
  /// The warped frame: the two cameras' six planes one after another, which
  /// is the (2, 6, H, W) the model takes. One allocation, so the staging
  /// reads it as one image.
  private let planes: UnsafeMutableRawPointer
  /// The NV12 each camera's access unit decodes into.
  private let roadNV12: UnsafeMutableRawPointer
  private let wideNV12: UnsafeMutableRawPointer

  /// The shape of the warped frame the model takes: (2, 6, H, W), the two
  /// cameras' six planes one after another. Nil when the shape is not one
  /// this can hold.
  init?(shape: [Int]) {
    guard shape.count == 4, shape.allSatisfy({ $0 > 0 }) else { return nil }
    // the two cameras, each a (6, H, W) set of planes
    guard shape[0] == 2, shape[1] == 6 else { return nil }
    planeHeight = shape[2]
    planeWidth = shape[3]
    // the NV12 the encoder took: twice as wide and twice as tall as a plane
    width = planeWidth * 2
    height = planeHeight * 2
    guard let roadDecoder = jl_hevc_create(Int32(width), Int32(height)) else { return nil }
    guard let wideDecoder = jl_hevc_create(Int32(width), Int32(height)) else {
      jl_hevc_free(roadDecoder)
      return nil
    }
    road = roadDecoder
    wide = wideDecoder
    let planeBytes = planeHeight * planeWidth
    // the two cameras' planes, one after another
    planes = .allocate(byteCount: 12 * planeBytes, alignment: 64)
    let frameBytes = jl_hevc_frame_bytes(Int32(width), Int32(height))
    roadNV12 = .allocate(byteCount: frameBytes, alignment: 64)
    wideNV12 = .allocate(byteCount: frameBytes, alignment: 64)
  }

  deinit {
    if let road { jl_hevc_free(road) }
    if let wide { jl_hevc_free(wide) }
    planes.deallocate()
    roadNV12.deallocate()
    wideNV12.deallocate()
  }

  /// The two access units at `src`, each behind its u32 length, into the
  /// six planes. False when the lengths do not add up to what is there, or
  /// when a stream owes an IDR it has not had: a frame that cannot be
  /// decoded is not a frame, and the caller says so rather than publishing
  /// pixels from the last one.
  func unpack(_ src: UnsafeRawBufferPointer) -> Bool {
    guard let base = src.baseAddress, src.count >= 8 else { return false }
    let roadLength = Int(base.loadUnaligned(fromByteOffset: 0, as: UInt32.self))
    let wideLength = Int(base.loadUnaligned(fromByteOffset: 4, as: UInt32.self))
    // the lengths are the whole of what is there: a length past the message
    // is caught here, not by a read past the end
    guard roadLength >= 0, wideLength >= 0, 8 + roadLength + wideLength == src.count else { return false }
    let roadBytes = UnsafeRawBufferPointer(rebasing: src[8..<(8 + roadLength)])
    let wideBytes = UnsafeRawBufferPointer(rebasing: src[(8 + roadLength)...])
    let planeBytes = planeHeight * planeWidth
    let roadPlanes = planes.advanced(by: 0)
    let widePlanes = planes.advanced(by: 6 * planeBytes)
    guard decode(roadBytes, into: roadNV12, planes: roadPlanes, decoder: road),
      decode(wideBytes, into: wideNV12, planes: widePlanes, decoder: wide)
    else { return false }
    return true
  }

  /// One access unit into its NV12 and its six planes.
  private func decode(_ bytes: UnsafeRawBufferPointer, into nv12: UnsafeMutableRawPointer,
                       planes: UnsafeMutableRawPointer, decoder: OpaquePointer?) -> Bool {
    guard let decoder, let base = bytes.baseAddress, bytes.count > 0 else { return false }
    var keyframe: Int32 = 0
    let got = nv12.withMemoryRebound(to: UInt8.self, capacity: jl_hevc_frame_bytes(Int32(width), Int32(height))) { out in
      jl_hevc_decode(decoder, base, bytes.count, out, &keyframe)
    }
    // 1 is a frame out; -2 is a stream that owes an IDR, which the caller
    // answers by waiting for the next one
    guard got == 1 else { return false }
    let planeBytes = planeHeight * planeWidth
    planes.withMemoryRebound(to: UInt8.self, capacity: 6 * planeBytes) { out in
      jl_hevc_unpack_nv12(nv12, Int32(width), Int32(height), out)
    }
    return true
  }

  /// The warped frame the model takes: the two cameras' six planes, one
  /// after another, (2, 6, H, W). Valid until the next `unpack`.
  var warped: UnsafeRawPointer {
    UnsafeRawPointer(planes)
  }

  /// Drops the reference pictures of both streams: the next access unit has
  /// to be a keyframe for anything to come out. The caller's answer to a
  /// gap (jetlink.openpilot.stream).
  func flush() {
    if let road { jl_hevc_flush(road) }
    if let wide { jl_hevc_flush(wide) }
  }

  /// Whether this build has a decoder at all: a host without libavcodec
  /// loaded cannot take an HEVC frame, and the hello must not offer it.
  static var available: Bool { jl_hevc_available() != 0 }

  /// The decoder's name, for a log line.
  static var backend: String { String(cString: jl_hevc_backend()) }
}
