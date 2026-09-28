import CTrt
import Foundation

/// Why TensorRT cannot run here: no driver, no TensorRT of the shim's major,
/// no such device, or the fake shim.
public struct TensorRTUnavailable: Error, CustomStringConvertible {
  public let description: String

  public init(_ description: String) {
    self.description = description
  }
}

/// TensorRT on this machine, as the shim finds it.
public enum TensorRT {
  /// Opens CUDA and TensorRT on `device` and closes them again. Returns what
  /// loaded, "TensorRT 10.16.2.10 on Orin (sm87)", or throws why nothing did.
  /// The fake shim always throws: a build without TensorRT's headers never
  /// runs a model on it.
  public static func probe(device: Int = 0) throws -> String {
    guard let index = Int32(exactly: device), index >= 0 else { throw TensorRTUnavailable("no CUDA device \(device)") }
    var trt: OpaquePointer?
    var err = [CChar](repeating: 0, count: 512)
    guard jl_trt_open(index, &trt, &err, err.count) == JL_TRT_OK, let trt else {
      throw TensorRTUnavailable(String(decoding: err.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self))
    }
    defer { jl_trt_close(trt) }
    var info = jl_trt_info()
    jl_trt_get_info(trt, &info)
    let device = info.device_name.map { String(cString: $0) } ?? "device \(index)"
    return "TensorRT \(info.major).\(info.minor).\(info.patch).\(info.build) on \(device) (sm\(info.cc_major)\(info.cc_minor))"
  }
}
