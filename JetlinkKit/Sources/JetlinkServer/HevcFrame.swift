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

  private let log = ServerLog(category: "hevc")
  /// When the layout line last went out. Once a second is enough to watch a
  /// carrier stay good or go bad, and costs nothing against a 20 Hz loop.
  private var lastReport = 0.0
  /// The same, for the line a refused frame writes.
  private var lastRefusal = 0.0
  /// The codec's own format, said once: what it decided the picture is.
  private var loggedFormat = false

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
    else {
      reportRefusal()
      return false
    }
    reportIfDue()
    return true
  }

  /// A frame this could not turn into planes, said once a second with the
  /// numbers that decide it. A rejected frame reaches the comma as BAD_SHAPE
  /// and hands the whole link back, so this line is what says why: a stream
  /// that owes an IDR, a buffer the codec laid out differently than it said,
  /// or a decoder that failed outright.
  private func reportRefusal() {
    let now = Date().timeIntervalSinceReferenceDate
    guard now - lastRefusal >= 1.0 else { return }
    lastRefusal = now
    guard let road, let info = jl_hevc_layout_of(road) else {
      log.warning("hevc refused a frame, and no decoder to explain it")
      return
    }
    let l = info.pointee
    log.warning("hevc refused a frame: \(width)x\(height)"
                + " stride=\(l.stride) slice=\(l.slice_height) fmt=0x\(String(l.color_format, radix: 16))"
                + " size=\(l.last_size) offset=\(l.last_offset) copies=\(l.copies)"
                + " strided=\(l.strided) tight=\(l.tight_fallback) refused=\(l.refused)")
  }

  /// One line a second: what the decoders say about their buffers, and what the
  /// first plane of each camera actually holds.
  ///
  /// This is the measurement that tells a picture broken by the carrier from a
  /// model that cannot read a picture. The comma logs the plane it is about to
  /// encode (`jetlink: warp frame ... narrow 65/5 moved 0.5`); if this line
  /// says `road 16/42` for the same second, the picture did not survive the
  /// carrier, and `stride`/`slice`/`fmt` say why. The four quadrant means catch
  /// a picture that fills only part of the buffer -- the other shape that reads
  /// as "mean drops, spread rises".
  private func reportIfDue() {
    let now = Date().timeIntervalSinceReferenceDate
    guard now - lastReport >= 1.0 else { return }
    lastReport = now
    var line = "hevc \(width)x\(height)"
    if let road, let info = jl_hevc_layout_of(road) {
      let l = info.pointee
      line += " stride=\(l.stride) slice=\(l.slice_height) fmt=0x\(String(l.color_format, radix: 16))"
      line += " copies=\(l.copies) strided=\(l.strided) tight=\(l.tight_fallback) refused=\(l.refused)"
      line += " size=\(l.last_size) off=\(l.last_offset) cap=\(l.out_cap)"
      line += " head=" + String(format: "%016llx%016llx", l.head_lo, l.head_hi)
    }
    if !loggedFormat, let road, let info = jl_hevc_layout_of(road) {
      var buffer = [CChar](repeating: 0, count: 512)
      let n = jl_hevc_format_string(road, &buffer, buffer.count)
      if n > 0 {
        loggedFormat = true
        log.warning("hevc codec format: " + String(cString: buffer))
      }
      _ = info
    }
    let planeBytes = planeHeight * planeWidth
    let roadStats = stats(planes)
    let wideStats = stats(planes.advanced(by: 6 * planeBytes))
    line += " | road \(roadStats.mean)/\(roadStats.spread) \(roadStats.quarters)"
    line += " even=\(roadStats.even) odd=\(roadStats.odd)"
    line += " wide \(wideStats.mean)/\(wideStats.spread) \(wideStats.quarters)"
    log.warning(line)
  }

  /// Mean, spread and four row-band means of one plane, sampled every 4th byte
  /// the way the trace's own fingerprints are.
  private func stats(_ at: UnsafeMutableRawPointer) -> (mean: Int, spread: Int, quarters: [Int],
                                                         even: Int, odd: Int) {
    let p = at.assumingMemoryBound(to: UInt8.self)
    var sum = 0.0, squares = 0.0, sampled = 0
    var evenSum = 0.0, evenCount = 0
    var oddSum = 0.0, oddCount = 0
    var quarters: [Int] = []
    // every 4th byte, and its neighbour across: a plane read as 8-bit gives
    // the two the same mean, a 10-bit one read as 8-bit splits them (low byte
    // against high), which is a bit depth problem wearing a shear's clothes
    for band in 0..<4 {
      let from = band * planeHeight / 4
      let to = (band + 1) * planeHeight / 4
      var bandSum = 0.0, bandCount = 0
      for y in from..<to {
        var x = 0
        while x + 1 < planeWidth {
          let i = y * planeWidth + x
          let a = Double(p[i])
          let b = Double(p[i + 1])
          sum += a
          squares += a * a
          sampled += 1
          bandSum += a
          bandCount += 1
          if i % 2 == 0 {
            evenSum += a
            evenCount += 1
            oddSum += b
            oddCount += 1
          } else {
            oddSum += a
            oddCount += 1
            evenSum += b
            evenCount += 1
          }
          x += 4
        }
      }
      quarters.append(bandCount > 0 ? Int((bandSum / Double(bandCount)).rounded()) : 0)
    }
    guard sampled > 0 else { return (0, 0, quarters, 0, 0) }
    let mean = sum / Double(sampled)
    let variance = max(squares / Double(sampled) - mean * mean, 0)
    let even = evenCount > 0 ? Int((evenSum / Double(evenCount)).rounded()) : 0
    let odd = oddCount > 0 ? Int((oddSum / Double(oddCount)).rounded()) : 0
    return (Int(mean.rounded()), Int(variance.squareRoot().rounded()), quarters, even, odd)
  }

  /// One access unit into its NV12 and its six planes.
  private func decode(
    _ bytes: UnsafeRawBufferPointer, into nv12: UnsafeMutableRawPointer, planes: UnsafeMutableRawPointer,
    decoder: OpaquePointer?
  ) -> Bool {
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
