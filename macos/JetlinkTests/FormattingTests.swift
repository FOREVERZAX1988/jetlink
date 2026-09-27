import Foundation
import JetlinkUI
import Testing

@testable import Jetlink

/// The text only the Mac's own views produce. What both apps show is tested in
/// JetlinkKit's FormattingTests.
@MainActor
@Suite("Formatting")
struct FormattingTests {
  @Test("The runtime line names the runtime, its version and the hardware")
  func runtimeLine() {
    #expect(StatusView.runtimeLine(backend: "tinygrad", version: "0.14.0+1241484386bc", device: "METAL-Apple_M1_Pro") == "tinygrad 0.14.0, Apple M1 Pro")
    #expect(StatusView.runtimeLine(backend: "ort", version: "1.29.0", device: "coreml-Apple_M1_Pro") == "onnxruntime 1.29.0, Apple M1 Pro")
  }

  // MARK: Status view text

  @Test("Backends are named the way the Settings picker names them")
  func backendDescription() {
    #expect(StatusView.backendDescription(backend: "ort", device: "coreml-Apple_M1_Pro") == "CoreML on the GPU")
    #expect(StatusView.backendDescription(backend: "ort", device: "ane-Apple_M1_Pro") == "CoreML with the Neural Engine")
    // what the removed Python server's tinygrad backend reported, named plainly
    #expect(StatusView.backendDescription(backend: "tinygrad", device: "METAL") == "tinygrad")
    #expect(StatusView.backendDescription(backend: "ort", device: "cpu-Apple_M1_Pro") == "onnxruntime on cpu-Apple_M1_Pro")
    #expect(StatusView.backendDescription(backend: "trt", device: "cuda") == "trt")
    #expect(StatusView.backendDescription(backend: nil, device: nil) == "Unknown")
  }

  @Test("An uptime under a minute says so instead of showing zero")
  func uptime() {
    let start = Date(timeIntervalSince1970: 1_757_440_000)
    #expect(StatusView.uptimeText(from: start, to: start) == "Less than a minute")
    #expect(StatusView.uptimeText(from: start, to: start.addingTimeInterval(59)) == "Less than a minute")
    #expect(StatusView.uptimeText(from: start, to: start.addingTimeInterval(60)) == "1 minute")
    #expect(StatusView.uptimeText(from: start, to: start.addingTimeInterval(150)) == "2 minutes")
    #expect(StatusView.uptimeText(from: start, to: start.addingTimeInterval(3900)) == "1 hour, 5 minutes")
  }

  @Test("The toolbar names the backend in a word or two")
  func backendShortName() {
    #expect(StatusView.backendShortName(backend: "ort", device: "ane-Apple_M1_Pro") == "Neural Engine")
    #expect(StatusView.backendShortName(backend: "ort", device: "coreml-Apple_M1_Pro") == "CoreML GPU")
    #expect(StatusView.backendShortName(backend: "tinygrad", device: "METAL-Apple_M1_Pro") == "tinygrad")
    #expect(BackendChoice.auto.shortTitle == "Neural Engine")
  }

  @Test("Log lines are coloured by their level")
  func logTone() {
    #expect(LogsView.tone(for: PreviewData.logLines[5]) == .red)
    #expect(LogsView.tone(for: PreviewData.logLines[3]) == .orange)
    #expect(LogsView.tone(for: PreviewData.logLines[0]) == .primary)
  }
}
