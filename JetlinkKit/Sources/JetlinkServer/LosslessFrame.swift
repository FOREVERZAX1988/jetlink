import CLossless
import Dispatch
import JetlinkKit

/// A lossless frame (`Wire.Flag.lossless`) back to the warped frame, bit for
/// bit: each plane's MED errors were packed alone with zstd
/// (jetlink.lossless), so the planes unpack in parallel, each on its own
/// decoder. The Python is `jetlink.protocol.LOSSLESS_*`.
///
/// Unchecked Sendable for `unpack`'s parallel loop: each iteration has its
/// own decoder, result slot and plane of `pixels`, and the host's lock lets
/// one frame unpack at a time.
final class LosslessFrame: @unchecked Sendable {
  static let codec = Pinned.losslessCodec

  let planes: Int
  let planeHeight: Int
  let planeWidth: Int
  /// The size table that leads the planes.
  var tableBytes: Int { planes * 4 }
  /// The warped frame, once `unpack` returns true.
  let pixels: UnsafeMutableRawPointer
  private let decoders: [OpaquePointer]
  private let results: UnsafeMutablePointer<Int32>
  /// Where each plane starts in the payload, and where the last one ends.
  private let offsets: UnsafeMutablePointer<Int>

  /// For a warped frame of shape (n, k, H, W): n * k planes of H x W. Nil
  /// for any other shape.
  init?(shape: [Int]) {
    guard shape.count == 4, shape.allSatisfy({ $0 > 0 }) else { return nil }
    planes = shape[0] * shape[1]
    planeHeight = shape[2]
    planeWidth = shape[3]
    var decoders: [OpaquePointer] = []
    for _ in 0..<planes {
      guard let decoder = jl_lossless_create() else {
        decoders.forEach { jl_lossless_free($0) }
        return nil
      }
      decoders.append(decoder)
    }
    self.decoders = decoders
    pixels = .allocate(byteCount: planes * planeHeight * planeWidth, alignment: 64)
    results = .allocate(capacity: planes)
    offsets = .allocate(capacity: planes + 1)
  }

  deinit {
    decoders.forEach { jl_lossless_free($0) }
    pixels.deallocate()
    results.deallocate()
    offsets.deallocate()
  }

  /// The size table and the planes at `src` into `pixels`. False when they
  /// are not this frame's planes: a table that does not add up to the bytes
  /// after it, or a plane that does not unpack to exactly one plane.
  func unpack(_ src: UnsafeRawBufferPointer) -> Bool {
    guard src.count >= tableBytes, let base = src.baseAddress else { return false }
    offsets[0] = tableBytes
    for k in 0..<planes {
      let size = Int(UInt32(littleEndian: base.loadUnaligned(fromByteOffset: k * 4, as: UInt32.self)))
      offsets[k + 1] = offsets[k] + size
    }
    guard offsets[planes] == src.count else { return false }
    let planeBytes = planeHeight * planeWidth
    let h = Int32(planeHeight), w = Int32(planeWidth)
    let pixels = pixels.assumingMemoryBound(to: UInt8.self)
    let offsets = offsets
    DispatchQueue.concurrentPerform(iterations: planes) { k in
      results[k] = jl_lossless_plane(
        decoders[k], base + offsets[k], offsets[k + 1] - offsets[k], h, w, pixels + k * planeBytes)
    }
    return (0..<planes).allSatisfy { results[$0] == 0 }
  }
}
