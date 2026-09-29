import Foundation
import JetlinkKit
import Testing

@testable import JetlinkServer

#if canImport(Glibc)
  import Glibc
#elseif canImport(Android)
  import Android
#endif

/// The host's USB read path over the fake usbfs, against a model of the
/// wire: the comma sends a 766 MB model's request (475,136 B padded) at the
/// 410 MB/s measured on the Jetson's link, and a packet moves only into a
/// read the host has posted, as the bus NAKs a device the host has not asked.
/// `span` is from the request's first byte on the wire to the whole request
/// in `recv`'s hands, and `cpu` the reading thread's CPU time per request.
/// `ring` keeps 16 KB reads posted; `turns` posts only what is asked, the
/// host's shape before the ring (header, 256 KB, the rest). The turns here
/// are thread wake-ups only: on a Jetson each also paid a USB 3 link power
/// state exit. Off unless JETLINK_BENCH is set, and only meaningful optimized:
///
///     JETLINK_BENCH=1 swift test -c release -Xswiftc -enable-testing --filter UsbfsRingBenchmark
@Suite("usbfs ring benchmark", .serialized, .enabled(if: ProcessInfo.processInfo.environment["JETLINK_BENCH"] != nil))
struct UsbfsRingBenchmark {
  static let requests = 500
  static let warmup = 50
  static let rate = 410e6
  static let payload = 475_136 - Wire.headerSize

  @Test("A 766 MB model's request, ring against turns")
  func request() throws {
    let wire = Double(UsbfsRingBenchmark.payload + Wire.headerSize) / UsbfsRingBenchmark.rate * 1e3
    print(String(format: "usbfs ring benchmark: %d requests of 475,136 B at 410 MB/s (wire %.3f ms)", UsbfsRingBenchmark.requests, wire))
    for aligned in [true, false] {
      let (spans, cpu, submits) = try UsbfsRingBenchmark.run(aligned: aligned)
      let sorted = spans.sorted()
      let at = { (q: Double) in sorted[min(sorted.count - 1, Int(Double(sorted.count) * q))] }
      print(
        String(
          format: "  %@  span p50 %.3f p99 %.3f max %.3f ms (%.3f over the wire)  cpu %.1f us  submits %.1f per request",
          aligned ? "ring " : "turns", at(0.5), at(0.99), sorted.last!, at(0.5) - wire, cpu, submits))
    }
  }

  static func run(aligned: Bool) throws -> (spans: [Double], cpu: Double, submits: Double) {
    let kernel = FakeUsbfs()
    let transport = USBTransport(pipes: UsbfsPipes(device: UsbfsDevice(kernel: kernel), inEndpoint: 0x81, outEndpoint: 0x01, aligned: aligned))
    let frame = FakePipes.gadgetFrame(.inferReq, seq: 1, payload: Data(pattern(payload)))
    let total = requests + warmup
    let starts = Samples(total)
    let received = DispatchSemaphore(value: 0)
    let comma = Thread {
      frame.withUnsafeBytes { bytes in
        for i in 0..<total {
          starts[i] = kernel.transmit(bytes, rate: rate)
          received.wait()
          // The server's model and the comma's own turn before the next one.
          Thread.sleep(forTimeInterval: 0.002)
        }
      }
    }
    comma.start()
    var spans: [Double] = []
    var cpu: UInt64 = 0
    var submits = 0
    for i in 0..<total {
      let before = threadCPU()
      let submitted = kernel.submits
      let message = try transport.recv()
      let done = DispatchTime.now().uptimeNanoseconds
      precondition(message.payload.count == payload)
      if i >= warmup {
        cpu += threadCPU() - before
        submits += kernel.submits - submitted
      }
      received.signal()
      if i >= warmup {
        spans.append(Double(done - starts[i]) / 1e6)
      }
    }
    return (spans, Double(cpu) / Double(requests) / 1e3, Double(submits) / Double(requests))
  }

  /// This thread's CPU time, in nanoseconds.
  static func threadCPU() -> UInt64 {
    var now = timespec()
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &now)
    return UInt64(now.tv_sec) * 1_000_000_000 + UInt64(now.tv_nsec)
  }
}

/// Timestamps one thread writes and another reads after a semaphore.
final class Samples: @unchecked Sendable {
  private var values: [UInt64]

  init(_ count: Int) {
    values = Array(repeating: 0, count: count)
  }

  subscript(index: Int) -> UInt64 {
    get { values[index] }
    set { values[index] = newValue }
  }
}
