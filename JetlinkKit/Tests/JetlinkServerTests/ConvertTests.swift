import Foundation
import Testing

@testable import JetlinkServer

/// The vImage conversions against the element loops they replaced, which are
/// what the Python queues compute: numpy's casts, round to nearest even. Bit
/// for bit, so NaN payloads, signs of zero and subnormals count.
@Suite("Convert")
struct ConvertTests {
  // The element loops, kept here as the reference.
  private static func referenceU8ToF16(_ src: [UInt8]) -> [UInt16] {
    src.map { Float16(Float($0)).bitPattern }
  }

  private static func referenceF32ToF16(_ src: [Float]) -> [UInt16] {
    src.map { Float16($0).bitPattern }
  }

  private static func referenceF16ToF32(_ src: [UInt16]) -> [UInt32] {
    src.map { Float(Float16(bitPattern: $0)).bitPattern }
  }

  /// Every float32 whose rounding to float16 is delicate: halfway cases both
  /// ways of the tie, the overflow edge, the subnormal range, both zeros, the
  /// infinities and NaNs with payloads, quiet and signaling.
  private static var edges: [Float] {
    var values: [Float] = [
      0, -0.0, 1, -1, 65504, -65504, 65520, 65519.99, 65536, 1e5, -1e5, 3.4e38, .infinity, -.infinity,
      .nan, -.nan, Float(bitPattern: 0x7FC0_0001), Float(bitPattern: 0x7F80_0001), Float(bitPattern: 0xFFC1_2345),
      Float(bitPattern: 0x7FBF_FFFF), .ulpOfOne, .leastNonzeroMagnitude, .leastNormalMagnitude, 1e-8, -1e-8,
    ]
    // Around one: exactly halfway between two float16s, and a hair either side.
    for k in 1...8 {
      let half = Float(k) * Float(Float16.ulpOfOne) / 2
      values += [1 + half, 1 + half.nextUp, 1 + half.nextDown, -(1 + half)]
    }
    // The float16 subnormals, 2^-24 apart, and the halfway points between them.
    let tiny = Float(Float16.leastNonzeroMagnitude)
    for k in 0...48 {
      let x = Float(k) * tiny
      values += [x, x + tiny / 2, (x + tiny / 2).nextUp, (x + tiny / 2).nextDown, -x - tiny / 2]
    }
    // The normal edge: 2^-14 and the largest subnormal.
    values += [Float(Float16.leastNormalMagnitude), Float(Float16.leastNormalMagnitude).nextDown, Float(Float16.greatestFiniteMagnitude).nextUp]
    return values
  }

  @Test("uint8 to float16 is the lookup, all 256 values")
  func u8ToF16() {
    let src = (0..<1024).map { UInt8(truncatingIfNeeded: $0 * 7 + $0 / 4) }
    var out = [UInt16](repeating: 0xFFFF, count: src.count)
    src.withUnsafeBytes { s in out.withUnsafeMutableBytes { d in Convert.u8ToF16(s.baseAddress!, d.baseAddress!, count: src.count) } }
    #expect(out == ConvertTests.referenceU8ToF16(src))
  }

  @Test("float32 to float16 rounds as the element loop, edges included")
  func f32ToF16() {
    var src = ConvertTests.edges
    var generator = SystemRandomNumberGenerator()
    for _ in 0..<100_000 {
      src.append(Float(bitPattern: UInt32.random(in: 0...UInt32.max, using: &generator)))
    }
    for _ in 0..<10_000 {
      src.append(Float.random(in: -70000...70000, using: &generator))
    }
    var out = [UInt16](repeating: 0xFFFF, count: src.count)
    src.withUnsafeBytes { s in out.withUnsafeMutableBytes { d in Convert.f32ToF16(s.baseAddress!, d.baseAddress!, count: src.count) } }
    let reference = ConvertTests.referenceF32ToF16(src)
    let differences = zip(out, reference).enumerated().filter { $0.element.0 != $0.element.1 }
    #expect(differences.isEmpty, "first difference at \(differences.first.map { "\($0.offset): \(src[$0.offset]) -> \($0.element)" } ?? "")")
  }

  @Test("float32 to float16 reads an unaligned source")
  func f32ToF16Unaligned() {
    let src = ConvertTests.edges
    var bytes = [UInt8](repeating: 0, count: src.count * 4 + 1)
    bytes.withUnsafeMutableBytes { b in
      src.withUnsafeBytes { s in b.baseAddress!.advanced(by: 1).copyMemory(from: s.baseAddress!, byteCount: s.count) }
    }
    var out = [UInt16](repeating: 0xFFFF, count: src.count)
    bytes.withUnsafeBytes { b in out.withUnsafeMutableBytes { d in Convert.f32ToF16(b.baseAddress! + 1, d.baseAddress!, count: src.count) } }
    #expect(out == ConvertTests.referenceF32ToF16(src))
  }

  @Test("float16 to float32 is exact for every float16 bit pattern")
  func f16ToF32() {
    let src = (0...UInt16.max).map { $0 }
    var out = [UInt32](repeating: 0, count: src.count)
    src.withUnsafeBytes { s in out.withUnsafeMutableBytes { d in Convert.f16ToF32(s.baseAddress!, d.baseAddress!, count: src.count) } }
    let reference = ConvertTests.referenceF16ToF32(src)
    let differences = zip(out, reference).enumerated().filter { $0.element.0 != $0.element.1 }
    #expect(differences.isEmpty, "first difference at \(differences.first.map { "0x\(String($0.offset, radix: 16)): \($0.element)" } ?? "")")
  }

  @Test("An empty conversion touches nothing")
  func empty() {
    var out: [UInt16] = [7]
    let src: [Float] = [1]
    src.withUnsafeBytes { s in out.withUnsafeMutableBytes { d in Convert.f32ToF16(s.baseAddress!, d.baseAddress!, count: 0) } }
    #expect(out == [7])
  }
}
