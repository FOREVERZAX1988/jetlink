import CTrt
import Foundation
import JetlinkTRT
import Testing

@Suite("TensorRT shim")
struct ShimTests {
  #if JL_TRT_FAKE
    @Test("The fake shim opens from Swift and reports what it models")
    func fakeOpens() throws {
      var trt: OpaquePointer?
      var err = [CChar](repeating: 0, count: 256)
      #expect(jl_trt_fake_open(nil, &trt, &err, err.count) == JL_TRT_OK)
      let handle = try #require(trt)
      defer { jl_trt_close(handle) }
      var info = jl_trt_info()
      jl_trt_get_info(handle, &info)
      #expect((info.major, info.minor, info.patch) == (10, 3, 0))
      #expect(String(cString: info.device_name) == "Orin" && (info.cc_major, info.cc_minor) == (8, 7))
      #expect(jl_trt_sticky(handle) == 0)
    }

    @Test("A binary built on the fake never finds TensorRT, and says so")
    func fakeIsUnavailable() {
      #expect(throws: TensorRTUnavailable.self) { try TensorRT() }
      #expect(throws: TensorRTUnavailable.self) { try TensorRT(device: -1) }
      #expect(throws: TensorRTUnavailable.self) { try TrtBackend() }
    }
  #endif
}
