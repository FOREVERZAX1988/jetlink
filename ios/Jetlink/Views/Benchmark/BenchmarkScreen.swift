import JetlinkKit
import JetlinkUI
import SwiftUI
import UIKit

/// Is this phone fast enough, and does it stay fast enough? The loaded model
/// at the comma's pace on the phone alone, then the commands that add the
/// cable from the comma and check the numbers from a Mac.
struct BenchmarkScreen: View {
  @Environment(AppModel.self) private var app
  @State private var refusal: String?
  @State private var starting = false
  /// `-benchmark 60` on the command line runs one once a model is loaded,
  /// for screenshots from the simulator.
  @State private var scripted: Double? = UserDefaults.standard.double(forKey: "benchmark") > 0 ? UserDefaults.standard.double(forKey: "benchmark") : nil

  var body: some View {
    NavigationStack {
      ScrollView {
        VStack(spacing: StatusContent.spacing) {
          runCard
          if let report {
            VerdictCard(report: report)
            totals(report)
            if report.windows.count > 1 {
              WindowsCard(windows: report.windows)
            }
          }
          SectionHeader("From the Comma")
          CommandCard(
            title: "Over the Cable", systemImage: "cable.connector", command: commaCommand,
            missing: "Load a model to get the command.",
            note:
              "Parked, with the cable in and Accelerator Link off on the comma, so its own client is not holding the port. Run it on the comma over SSH: it waits for the phone to dial, then sends 1,200 real-sized frames at 20 a second and reports the round trip the car will see. Frames over 50 ms should be 0."
          )
          SectionHeader("Accuracy")
          CommandCard(
            title: "From a Mac", systemImage: "checkmark.seal", command: parityCommand,
            missing: app.network.wifi == nil ? "Join Wi-Fi and load a model to get the command." : "Load a model to get the command.",
            note:
              "Checks that this phone computes what onnxruntime does on a computer, the gate the Mac server passes. Run it in a jetlink checkout on a Mac on the same Wi-Fi; it reads the model where the Mac app keeps it. Wi-Fi is slow, and this test does not care. It should end with OK."
          )
        }
        .padding(.bottom, 24)
      }
      .contentMargins(.horizontal, StatusContent.margin, for: .scrollContent)
      .background(Color.groupedBackground)
      .navigationTitle("Benchmark")
      .onChange(of: sha256, initial: true) {
        if let seconds = scripted, blocker == nil {
          scripted = nil
          start(seconds: seconds)
        }
      }
      .toolbar {
        if let report {
          ToolbarItem(placement: .topBarTrailing) {
            ShareLink(item: report.text, preview: SharePreview("Jetlink benchmark"))
          }
        }
      }
    }
  }

  // MARK: state

  private var event: BenchmarkEvent? { app.server.benchmark }
  private var running: Bool { event.map { !$0.isFinished } ?? false }
  private var report: BenchmarkReport? { event?.report }
  private var sha256: String? { app.server.engine.state == .ready ? app.server.engine.sha256 : nil }
  private var connected: Bool { app.server.link.state == .connected }

  private var modelBytes: Int64? {
    guard let sha256 else { return nil }
    return app.models.inventory?.models.first { $0.sha256 == sha256 }?.bytes ?? app.models.rows.first { $0.sha256 == sha256 }?.bytes
  }

  /// Why a run cannot start now, in a few words; nil when it can.
  private var blocker: String? {
    if app.server.runState != .serving { return "The server is not running." }
    if sha256 == nil { return "Load a model first." }
    if connected { return "Disconnect the comma to benchmark. Its live numbers are on Status." }
    return nil
  }

  // MARK: the run

  private var runCard: some View {
    SummaryCard(title: "Benchmark", systemImage: "stopwatch.fill", tint: .blue, trailing: app.settings.device.title) {
      VStack(alignment: .leading, spacing: 14) {
        Text(sha256.flatMap(app.modelName) ?? (sha256 == nil ? "No Model" : "Model"))
          .font(.title2.weight(.bold))
        if let event, running {
          progress(event)
        } else {
          buttons
        }
        if let text = refusal ?? (event?.state == "failed" ? event?.detail : nil) {
          Text(text)
            .font(.footnote)
            .foregroundStyle(.red)
        } else if !running, let blocker {
          Text(blocker)
            .font(.footnote)
            .foregroundStyle(.orange)
        }
        Text(
          "Runs the loaded model 20 times a second, as the comma will, on made-up camera frames: the history queues, the model, and reading the answer back. The cable is not in it. Leave the phone as it will be in the car, charging and mounted."
        )
        .font(.footnote)
        .foregroundStyle(.secondary)
      }
    }
  }

  private var buttons: some View {
    HStack(spacing: 12) {
      Button {
        start(seconds: 60)
      } label: {
        Label("1 Minute", systemImage: "play.fill")
          .frame(maxWidth: .infinity)
      }
      .buttonStyle(.glassProminent)
      Button {
        start(seconds: 600)
      } label: {
        Label("10 Minutes", systemImage: "flame.fill")
          .frame(maxWidth: .infinity)
      }
      .buttonStyle(.glass)
    }
    .controlSize(.large)
    .disabled(blocker != nil || starting)
  }

  private func progress(_ event: BenchmarkEvent) -> some View {
    VStack(alignment: .leading, spacing: 12) {
      ProgressView(value: min(event.elapsed / max(event.total, 1), 1))
        .tint(.blue)
      HStack {
        Text("\(BenchmarkScreen.clock(event.elapsed)) of \(BenchmarkScreen.clock(event.total))")
        Spacer()
        Text("\(event.frames.formatted()) frames")
      }
      .font(.subheadline.monospacedDigit())
      .foregroundStyle(.secondary)
      HStack(spacing: 0) {
        Figure("P50", ms: event.frame?.p50)
        Divider().frame(height: 32)
        Figure("P99", ms: event.frame?.p99)
      }
      Button(role: .destructive) {
        cancel()
      } label: {
        Label("Cancel", systemImage: "stop.fill")
          .frame(maxWidth: .infinity)
      }
      .buttonStyle(.glass)
      .controlSize(.large)
    }
  }

  private func start(seconds: Double) {
    refusal = nil
    starting = true
    Task {
      defer { starting = false }
      do {
        let reply = try await app.server.send(.benchmark(seconds: seconds))
        if !reply.ok { refusal = reply.error ?? "The benchmark could not start." }
      } catch {
        refusal = error.localizedDescription
      }
    }
  }

  private func cancel() {
    Task { _ = try? await app.server.send(.cancelBenchmark) }
  }

  /// "1:00" for 60 seconds.
  static func clock(_ seconds: Double) -> String {
    let whole = Int(seconds.rounded(.down))
    return "\(whole / 60):\(String(format: "%02d", whole % 60))"
  }

  // MARK: totals

  private func totals(_ report: BenchmarkReport) -> some View {
    MetricGrid {
      GridRow {
        MetricTile(
          title: "Frames", systemImage: "film.stack", tint: .teal, value: report.frames.formatted(),
          note: "\(BenchmarkScreen.clock(report.seconds)) at 20 Hz")
        MetricTile(
          title: "Over Budget", systemImage: "tortoise.fill", tint: .pink, value: report.over50.formatted(),
          note: report.over35 > 0 ? "\(report.over35.formatted()) over 35 ms" : "None over 35 ms", noteTone: report.over50 > 0 ? .red : nil)
      }
      GridRow {
        MetricTile(
          title: "Temperature", systemImage: DeviceHealth.Thermal(label: report.thermalAtEnd).symbol, tint: .orange,
          value: DeviceHealth.Thermal(label: report.thermalAtEnd).title,
          note: report.thermalAtStart == report.thermalAtEnd ? "Throughout" : "From \(DeviceHealth.Thermal(label: report.thermalAtStart).title.lowercased())",
          noteTone: DeviceHealth.Thermal(label: report.thermalAtEnd).tone)
        MetricTile(
          title: "Model", systemImage: "cpu", tint: .purple, value: report.accelerator.mean.formatted(.number.precision(.fractionLength(1))), unit: "ms",
          note: "Mean, model alone")
      }
    }
  }

  // MARK: commands

  /// bench_link on the comma, listening for this phone's dial over the cable.
  private var commaCommand: String? {
    guard let sha256, let bytes = modelBytes else { return nil }
    let listen = "--listen 0.0.0.0:\(PhoneServer.commaDial.port)"
    let model = "--sha256 \(sha256) --nbytes \(bytes)"
    return "cd /data/openpilot/jetlink_repo && PYTHONPATH=/data/openpilot python3 scripts/bench_link.py \(listen) \(model) --rate 20 --n 1200"
  }

  /// verify_parity from a Mac on the same Wi-Fi, dialing the phone's listener.
  private var parityCommand: String? {
    guard let sha256, let bytes = modelBytes, let wifi = app.network.wifi, let port = app.server.port else { return nil }
    let onnx = "\"$HOME/Library/Application Support/Jetlink/cache/models/\(sha256.prefix(16)).onnx\""
    return """
      python3 scripts/verify_parity.py capture --host \(wifi.address) --port \(port) --sha256 \(sha256) --nbytes \(bytes) --dir parity-iphone \\
        && python3 scripts/verify_parity.py reference --onnx \(onnx) --dir parity-iphone \\
        && python3 scripts/verify_parity.py compare --dir parity-iphone
      """
  }
}

/// Fast enough, tight, or too slow, and the numbers that say so.
struct VerdictCard: View {
  let report: BenchmarkReport

  enum Verdict {
    case good, tight, slow

    init(_ report: BenchmarkReport) {
      if report.frame.p99 <= 35 && report.over50 == 0 {
        self = .good
      } else if report.frame.p99 <= 50 {
        self = .tight
      } else {
        self = .slow
      }
    }

    var title: String {
      switch self {
      case .good: "Fast Enough"
      case .tight: "Tight"
      case .slow: "Too Slow"
      }
    }

    var detail: String {
      switch self {
      case .good: "Room for the cable in the 50 ms budget."
      case .tight: "Little room left for the cable."
      case .slow: "Misses 20 frames a second."
      }
    }

    var symbol: String {
      switch self {
      case .good: "checkmark.seal.fill"
      case .tight: "exclamationmark.triangle.fill"
      case .slow: "xmark.octagon.fill"
      }
    }

    var tone: Color {
      switch self {
      case .good: .green
      case .tight: .orange
      case .slow: .red
      }
    }
  }

  var body: some View {
    let verdict = Verdict(report)
    SummaryCard(title: "Verdict", systemImage: verdict.symbol, tint: verdict.tone, trailing: report.cancelled ? "Stopped Early" : nil) {
      VStack(alignment: .leading, spacing: 14) {
        VStack(alignment: .leading, spacing: 4) {
          Text(verdict.title)
            .font(.system(.title, design: .rounded, weight: .bold))
            .foregroundStyle(verdict.tone)
          Text(verdict.detail)
            .font(.subheadline)
            .foregroundStyle(.secondary)
        }
        HStack(spacing: 0) {
          Figure("P99", ms: report.frame.p99, tone: verdict.tone)
          Divider().frame(height: 32)
          Figure("Max", ms: report.frame.max)
          Divider().frame(height: 32)
          Figure("Mean", ms: report.frame.mean)
        }
      }
    }
  }
}

/// The run ten seconds at a time, with the phone's temperature as each closed.
struct WindowsCard: View {
  let windows: [BenchmarkWindow]

  var body: some View {
    SummaryCard(title: "Windows", systemImage: "chart.bar.fill", tint: .indigo, trailing: "10 s each") {
      VStack(spacing: 0) {
        ForEach(windows, id: \.startSecond) { window in
          let thermal = DeviceHealth.Thermal(label: window.thermal)
          HStack {
            Text(BenchmarkScreen.clock(Double(window.startSecond)))
              .foregroundStyle(.secondary)
              .frame(width: 44, alignment: .leading)
            Text("P99 \(window.frame.p99.formatted(.number.precision(.fractionLength(1)))) ms")
              .foregroundStyle(window.frame.p99 > 50 ? .red : (window.frame.p99 > 35 ? .orange : .primary))
            Spacer()
            Label(thermal.title, systemImage: thermal.symbol)
              .foregroundStyle(thermal.tone)
              .labelStyle(.titleAndIcon)
          }
          .font(.subheadline.monospacedDigit())
          .padding(.vertical, 6)
          if window.startSecond != windows.last?.startSecond {
            Divider()
          }
        }
      }
    }
  }
}

/// A shell command with a Copy button, or why there is none yet.
struct CommandCard: View {
  let title: String
  let systemImage: String
  let command: String?
  let missing: String
  let note: String
  @State private var copied = false

  var body: some View {
    SummaryCard(title: title, systemImage: systemImage, tint: .gray) {
      VStack(alignment: .leading, spacing: 12) {
        if let command {
          Text(command)
            .font(.caption.monospaced())
            .textSelection(.enabled)
            .frame(maxWidth: .infinity, alignment: .leading)
          Button {
            UIPasteboard.general.string = command
            copied = true
          } label: {
            Label(copied ? "Copied" : "Copy", systemImage: copied ? "checkmark" : "doc.on.doc")
          }
          .buttonStyle(.glass)
        } else {
          Text(missing)
            .font(.subheadline)
            .foregroundStyle(.secondary)
        }
        Text(note)
          .font(.footnote)
          .foregroundStyle(.secondary)
      }
    }
    .onChange(of: command) { copied = false }
  }
}
