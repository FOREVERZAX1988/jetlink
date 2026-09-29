import CTrt
import Foundation
import JetlinkTRT
import Testing

@Suite("TensorRT shim")
struct ShimTests {
  #if JL_TRT_FAKE
    @Test("A binary built on the fake never finds TensorRT, and says so")
    func fakeIsUnavailable() {
      for device in [0, -1] {
        #expect {
          try TensorRT(device: device)
        } throws: {
          ($0 as? TrtError)?.code == Int32(JL_TRT_UNAVAILABLE)
        }
      }
    }
  #endif
}
