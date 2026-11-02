import CLossless
import Foundation
import Testing

@testable import JetlinkServer

/// The MED errors of one plane, folded, as jetlink.lossless.encode makes them.
private func encode(_ plane: [UInt8], h: Int, w: Int) -> [UInt8] {
  func at(_ i: Int, _ j: Int) -> Int { i < 0 || j < 0 ? 0 : Int(plane[i * w + j]) }
  var out = [UInt8](repeating: 0, count: h * w)
  for i in 0..<h {
    for j in 0..<w {
      let a = at(i, j - 1), b = at(i - 1, j), c = at(i - 1, j - 1)
      let pred = c >= max(a, b) ? min(a, b) : c <= min(a, b) ? max(a, b) : a + b - c
      let r = ((Int(plane[i * w + j]) - pred) % 256 + 256) % 256
      out[i * w + j] = UInt8(r < 128 ? r * 2 : (256 - r) * 2 - 1)
    }
  }
  return out
}

/// A zstd frame holding `bytes` in one raw block: single segment, its size in
/// one byte (under 256), no checksum. Any zstd decoder reads it.
private func rawZstdFrame(_ bytes: [UInt8]) -> [UInt8] {
  precondition(bytes.count < 256)
  let block = 1 | (bytes.count << 3)  // last block, raw
  return [0x28, 0xB5, 0x2F, 0xFD, 0x20, UInt8(bytes.count), UInt8(block & 0xFF), UInt8((block >> 8) & 0xFF), UInt8(block >> 16)] + bytes
}

private func planes(_ shape: [Int], seed: UInt64) -> [[UInt8]] {
  var state = seed
  func next() -> UInt8 {
    state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
    return UInt8(truncatingIfNeeded: state >> 33)
  }
  let size = shape[2] * shape[3]
  return (0..<(shape[0] * shape[1])).map { k in
    // noise for half the planes, a gradient for the rest: every wrap, and runs
    k % 2 == 0 ? (0..<size).map { _ in next() } : (0..<size).map { UInt8(truncatingIfNeeded: $0 * 7 + k) }
  }
}

private func payload(_ planes: [[UInt8]], h: Int, w: Int) -> [UInt8] {
  let frames = planes.map { rawZstdFrame(encode($0, h: h, w: w)) }
  var out: [UInt8] = []
  for frame in frames {
    withUnsafeBytes(of: UInt32(frame.count).littleEndian) { out += $0 }
  }
  return out + frames.flatMap { $0 }
}

@Suite("Lossless frames")
struct LosslessFrameTests {
  let shape = [2, 6, 8, 16]

  @Test("The MED inverse gives every plane back")
  func unmed() {
    for plane in planes(shape, seed: 1) {
      let errors = encode(plane, h: 8, w: 16)
      var out = [UInt8](repeating: 0, count: plane.count)
      jl_lossless_unmed(errors, 8, 16, &out)
      #expect(out == plane)
    }
  }

  @Test("A frame's planes unpack in parallel, bit for bit")
  func unpack() throws {
    let frame = try #require(LosslessFrame(shape: shape))
    let source = planes(shape, seed: 2)
    let bytes = payload(source, h: 8, w: 16)
    let ok = bytes.withUnsafeBytes { frame.unpack($0) }
    #expect(ok)
    let got = Array(UnsafeBufferPointer(start: frame.pixels.assumingMemoryBound(to: UInt8.self), count: 12 * 128))
    #expect(got == source.flatMap { $0 })
  }

  @Test("A table that does not add up, or a plane that is not one, is refused")
  func refused() throws {
    let frame = try #require(LosslessFrame(shape: shape))
    let bytes = payload(planes(shape, seed: 3), h: 8, w: 16)
    #expect(!Array(bytes.dropLast()).withUnsafeBytes { frame.unpack($0) })
    #expect(!Array(bytes.prefix(12 * 4 - 1)).withUnsafeBytes { frame.unpack($0) })
    var badMagic = bytes
    badMagic[12 * 4] ^= 0xFF
    #expect(!badMagic.withUnsafeBytes { frame.unpack($0) })
    // a plane one pixel short of a plane
    let short = planes(shape, seed: 4).map { Array($0.dropLast()) }
    var shortPayload: [UInt8] = []
    let frames = short.map { rawZstdFrame($0) }
    for f in frames {
      withUnsafeBytes(of: UInt32(f.count).littleEndian) { shortPayload += $0 }
    }
    shortPayload += frames.flatMap { $0 }
    #expect(!shortPayload.withUnsafeBytes { frame.unpack($0) })
  }

  @Test("Only an (n, k, H, W) shape makes a frame")
  func shapes() {
    #expect(LosslessFrame(shape: [12, 8, 16]) == nil)
    #expect(LosslessFrame(shape: [2, 6, 0, 16]) == nil)
    #expect(LosslessFrame(shape: shape)?.planes == 12)
  }
}
