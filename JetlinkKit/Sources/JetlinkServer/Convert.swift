#if canImport(Accelerate)
  import Accelerate
#endif

/// The frame path's bulk conversions, through Accelerate's vImage.
///
/// A frame converts about 800,000 camera bytes to float16 and 35,000 floats
/// each way. As element loops these cost 0.2 ms in an optimized build and
/// over 40 ms in an unoptimized one: an iPhone 17 Pro running the app as
/// Xcode's Run builds it, Debug, measured 65 ms a frame around a 21 ms model.
/// vImage is vectorized whatever the build, and ConvertTests holds each of
/// these to the element loop bit for bit, NaN, overflow and subnormals
/// included, so the golden server tests keep matching the Python queues.
///
/// Where there is no Accelerate (the Linux build the conformance suite runs
/// on), the element loops themselves stand in: they are the reference.
enum Convert {
  /// uint8 to float16 bit patterns: 0...255 are all exact in float16.
  private static let u8ToF16Bits: [UInt16] = (0..<256).map { Float16(Float($0)).bitPattern }

  #if canImport(Accelerate)
    private static func buffer(_ pointer: UnsafeRawPointer, _ count: Int, _ size: Int) -> vImage_Buffer {
      vImage_Buffer(data: UnsafeMutableRawPointer(mutating: pointer), height: 1, width: vImagePixelCount(count), rowBytes: count * size)
    }

    static func u8ToF16(_ source: UnsafeRawPointer, _ destination: UnsafeMutableRawPointer, count: Int) {
      guard count > 0 else { return }
      var src = buffer(source, count, 1)
      var dst = buffer(destination, count, 2)
      u8ToF16Bits.withUnsafeBufferPointer { table in
        _ = vImageLookupTable_Planar8toPlanar16(&src, &dst, table.baseAddress!, vImage_Flags(kvImageDoNotTile))
      }
    }

    /// Round to nearest even, as numpy's float32 to float16 cast. The source
    /// need not be aligned.
    static func f32ToF16(_ source: UnsafeRawPointer, _ destination: UnsafeMutableRawPointer, count: Int) {
      guard count > 0 else { return }
      var src = buffer(source, count, 4)
      var dst = buffer(destination, count, 2)
      _ = vImageConvert_PlanarFtoPlanar16F(&src, &dst, vImage_Flags(kvImageDoNotTile))
    }

    static func f16ToF32(_ source: UnsafeRawPointer, _ destination: UnsafeMutableRawPointer, count: Int) {
      guard count > 0 else { return }
      var src = buffer(source, count, 2)
      var dst = buffer(destination, count, 4)
      _ = vImageConvert_Planar16FtoPlanarF(&src, &dst, vImage_Flags(kvImageDoNotTile))
    }
  #else
    static func u8ToF16(_ source: UnsafeRawPointer, _ destination: UnsafeMutableRawPointer, count: Int) {
      let out = destination.assumingMemoryBound(to: UInt16.self)
      u8ToF16Bits.withUnsafeBufferPointer { table in
        for i in 0..<count { out[i] = table[Int(source.load(fromByteOffset: i, as: UInt8.self))] }
      }
    }

    static func f32ToF16(_ source: UnsafeRawPointer, _ destination: UnsafeMutableRawPointer, count: Int) {
      let out = destination.assumingMemoryBound(to: UInt16.self)
      for i in 0..<count { out[i] = Float16(source.loadUnaligned(fromByteOffset: i * 4, as: Float.self)).bitPattern }
    }

    static func f16ToF32(_ source: UnsafeRawPointer, _ destination: UnsafeMutableRawPointer, count: Int) {
      let out = destination.assumingMemoryBound(to: Float.self)
      for i in 0..<count { out[i] = Float(Float16(bitPattern: source.loadUnaligned(fromByteOffset: i * 2, as: UInt16.self))) }
    }
  #endif
}
