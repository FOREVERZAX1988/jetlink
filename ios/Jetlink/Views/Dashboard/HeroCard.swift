import JetlinkKit
import JetlinkUI
import SwiftUI

/// The big card: the frame budget while the comma is being served, and
/// otherwise whatever stands between the comma and the big model.
struct HeroCard: View {
  let state: DashboardState
  /// For a phone on its side: the ring sized to the height, no caption.
  var compact = false
  var onUseDefault: () -> Void = {}
  var onOpenModels: () -> Void = {}
  var onRetry: () -> Void = {}
  var onOpenSettings: () -> Void = {}

  var body: some View {
    Card {
      switch state.hero {
      case .budget(let stats):
        budget(stats)
      case .progress(let engine):
        progress(engine)
      case .waiting:
        waiting
      case .noModel:
        noModel
      case .failed(let detail):
        failed(detail)
      }
    }
  }

  // MARK: serving

  private func budget(_ stats: StatsEvent?) -> some View {
    VStack(spacing: compact ? 8 : 14) {
      HeadroomRing(p99: stats?.served.p99, lineWidth: compact ? 14 : 18)
        .frame(maxWidth: 320, maxHeight: compact ? 190 : .infinity)
        .frame(maxWidth: .infinity)
      if let stats {
        HStack {
          figure("Mean", stats.served.mean)
          Divider().frame(height: 30)
          figure("p99", stats.served.p99)
          Divider().frame(height: 30)
          figure("Worst", stats.served.max)
        }
      }
      if !compact {
        Text("Round trip on this iPhone over the last ten seconds, from a frame's arrival to its reply leaving.")
          .font(.caption)
          .foregroundStyle(.secondary)
          .multilineTextAlignment(.center)
          .frame(maxWidth: .infinity)
      }
    }
  }

  private func figure(_ label: String, _ ms: Double) -> some View {
    VStack(spacing: 2) {
      Text(label)
        .font(.caption)
        .foregroundStyle(.secondary)
      Text(FrameBudgetView.ms(ms))
        .font(.headline)
        .contentTransition(.numericText(value: ms))
    }
    .frame(maxWidth: .infinity)
  }

  // MARK: preparing

  private func progress(_ engine: EngineEvent) -> some View {
    VStack(spacing: 16) {
      ZStack {
        Circle()
          .stroke(Color.blue.opacity(0.18), lineWidth: 14)
        Circle()
          .trim(from: 0, to: min(max(engine.frac, 0.02), 1))
          .stroke(Color.blue, style: StrokeStyle(lineWidth: 14, lineCap: .round))
          .rotationEffect(.degrees(-90))
          .animation(.smooth, value: engine.frac)
        VStack(spacing: 2) {
          Text(engine.frac.formatted(.percent.precision(.fractionLength(0))))
            .font(.system(size: 44, weight: .semibold))
            .contentTransition(.numericText(value: engine.frac))
          Text(ProgressRow.stageName(engine.stage))
            .font(.headline)
            .foregroundStyle(.secondary)
        }
      }
      .frame(width: 200, height: 200)
      .frame(maxWidth: .infinity)
      if !engine.msg.isEmpty {
        Text(DashboardState.sentence(engine.msg))
          .font(.subheadline)
          .foregroundStyle(.secondary)
          .multilineTextAlignment(.center)
          .frame(maxWidth: .infinity)
      }
      Text("Keep Jetlink open until this finishes.")
        .font(.caption)
        .foregroundStyle(.secondary)
        .multilineTextAlignment(.center)
        .frame(maxWidth: .infinity)
    }
    .accessibilityElement(children: .combine)
  }

  // MARK: waiting

  private var waiting: some View {
    VStack(spacing: 14) {
      Image(systemName: "cable.connector")
        .font(.system(size: 52, weight: .regular))
        .foregroundStyle(.secondary)
        .symbolEffect(.pulse, options: .repeating)
        .padding(.top, 8)
      Text(state.link.state == .disconnected ? "Waiting for the comma to reconnect" : "Waiting for the comma")
        .font(.title3.weight(.semibold))
        .multilineTextAlignment(.center)
      if state.localNetworkDenied {
        Text("Turn on Local Network for Jetlink in the Settings app, under Privacy & Security, or the comma's connection is refused.")
          .font(.subheadline)
          .foregroundStyle(.secondary)
          .multilineTextAlignment(.center)
        Button("Open Settings", action: onOpenSettings)
          .buttonStyle(.glassProminent)
      } else if let endpoint = state.endpoint {
        VStack(spacing: 4) {
          Text("The comma's JetlinkEndpoint")
            .font(.subheadline)
            .foregroundStyle(.secondary)
          Text(endpoint)
            .font(.system(.title2, design: .monospaced).weight(.medium))
            .textSelection(.enabled)
        }
      } else {
        Text("Connect this iPhone to the comma with a USB-C Ethernet adapter, then set the comma's JetlinkEndpoint to this iPhone's address.")
          .font(.subheadline)
          .foregroundStyle(.secondary)
          .multilineTextAlignment(.center)
      }
    }
    .frame(maxWidth: .infinity)
    .padding(.bottom, 8)
  }

  // MARK: nothing prepared

  private var noModel: some View {
    VStack(spacing: 14) {
      Image(systemName: "shippingbox")
        .font(.system(size: 48))
        .foregroundStyle(.secondary)
        .padding(.top, 8)
      Text("No model on this iPhone")
        .font(.title3.weight(.semibold))
      Text("Get the model your comma uses now, while there is Wi-Fi. Otherwise the comma sends its model when it connects, and drives on its small model until the iPhone is ready.")
        .font(.subheadline)
        .foregroundStyle(.secondary)
        .multilineTextAlignment(.center)
      if let row = state.defaultModel, ModelStore.canUse(row) {
        Button(action: onUseDefault) {
          Text("Use \(row.displayName)")
            .frame(maxWidth: .infinity)
        }
        .buttonStyle(.glassProminent)
        .controlSize(.large)
      }
      Button("Choose Another Model", action: onOpenModels)
    }
    .frame(maxWidth: .infinity)
  }

  // MARK: failed

  private func failed(_ detail: String) -> some View {
    VStack(spacing: 12) {
      Image(systemName: "exclamationmark.triangle.fill")
        .font(.system(size: 44))
        .foregroundStyle(.red)
        .padding(.top, 8)
      Text(state.runState == .serving ? "The model failed" : "The server stopped")
        .font(.title3.weight(.semibold))
      Text(detail)
        .font(.footnote.monospaced())
        .foregroundStyle(.secondary)
        .multilineTextAlignment(.center)
        .lineLimit(6)
        .textSelection(.enabled)
      Button("Try Again", action: onRetry)
        .buttonStyle(.glass)
    }
    .frame(maxWidth: .infinity)
  }
}
