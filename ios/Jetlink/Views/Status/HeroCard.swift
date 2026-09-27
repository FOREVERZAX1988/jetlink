import JetlinkKit
import JetlinkUI
import SwiftUI

/// What the Status tab's actions do, handed down from the screen.
struct StatusActions {
  var useDefault: () -> Void = {}
  var openModels: () -> Void = {}
  var retry: () -> Void = {}
  var openSettings: () -> Void = {}
  var refreshCatalog: () -> Void = {}
  var copyEndpoint: (String) -> Void = { _ in }
}

/// The main card: the headroom while the comma is served, and otherwise
/// whatever stands between the comma and the big model.
struct HeroCard: View {
  let state: StatusState
  var compact = false
  var actions = StatusActions()

  var body: some View {
    switch state.hero {
    case .budget(let stats):
      headroom(stats)
    case .progress(let engine):
      progress(engine)
    case .waiting:
      if state.localNetworkDenied {
        blocked
      } else {
        waiting
      }
    case .noModel:
      noModel
    case .failed(let detail):
      failed(detail)
    }
  }

  // MARK: serving

  private func headroom(_ stats: StatsEvent?) -> some View {
    let room = stats.map { FrameBudgetView.Room(headroomMs: FrameBudgetView.budgetMs - $0.servedMs.p99) }
    return SummaryCard(title: "Headroom", systemImage: "gauge.with.needle.fill", tint: room?.tone.color ?? .secondary, trailing: "Last 10 s") {
      VStack(spacing: compact ? 12 : 20) {
        HeadroomRing(p99: stats?.servedMs.p99, lineWidth: compact ? 16 : 22)
          .frame(maxWidth: compact ? 210 : 290)
          .frame(maxWidth: .infinity)
        HStack(spacing: 0) {
          figure("P99", stats?.servedMs.p99)
          Divider().frame(height: 32)
          figure("Max", stats?.servedMs.max)
        }
      }
    }
  }

  private func figure(_ label: String, _ ms: Double?) -> some View {
    VStack(spacing: 2) {
      Text(label)
        .font(.footnote.weight(.medium))
        .foregroundStyle(.secondary)
      HStack(alignment: .firstTextBaseline, spacing: 2) {
        Text(ms.map { $0.formatted(.number.precision(.fractionLength(1))) } ?? "--")
          .font(.system(.title3, design: .rounded, weight: .semibold))
          .contentTransition(.numericText(value: ms ?? 0))
        Text("ms")
          .font(.system(.footnote, design: .rounded, weight: .semibold))
          .foregroundStyle(.secondary)
      }
    }
    .frame(maxWidth: .infinity)
    .accessibilityElement(children: .combine)
  }

  // MARK: preparing

  private func progress(_ engine: EngineEvent) -> some View {
    let title = engine.state == .loading ? "Loading" : "Preparing"
    return SummaryCard(title: title, systemImage: "gearshape.2.fill", tint: .blue, trailing: engine.frac.formatted(.percent.precision(.fractionLength(0)))) {
      VStack(alignment: .leading, spacing: 10) {
        Text(state.modelName ?? "Model")
          .font(.title2.weight(.bold))
        ProgressView(value: min(max(engine.frac, 0), 1))
          .tint(.blue)
        Text(ProgressRow.stageName(engine.stage))
          .font(.subheadline)
          .foregroundStyle(.secondary)
          .contentTransition(.opacity)
      }
    }
  }

  // MARK: waiting

  private var waiting: some View {
    card {
      ContentUnavailableView {
        Label("Waiting for Comma", systemImage: "cable.connector")
          .symbolEffect(.pulse, options: .repeating)
      } description: {
        if state.cableAddress != nil {
          Text("Dialing the comma over USB.")
        } else {
          Text(state.endpoint == nil ? "Plug in the comma." : "Over USB, nothing to set. Over Ethernet, set the comma's endpoint to this address.")
        }
      } actions: {
        if state.cableAddress == nil, let endpoint = state.endpoint {
          Button {
            actions.copyEndpoint(endpoint)
          } label: {
            Label(endpoint, systemImage: "doc.on.doc")
              .font(.body.monospacedDigit().weight(.medium))
          }
          .buttonStyle(.glass)
        }
      }
    }
  }

  private var blocked: some View {
    card {
      ContentUnavailableView {
        Label("Local Network Off", systemImage: "wifi.exclamationmark")
      } description: {
        Text("Allow Local Network access so the comma can connect.")
      } actions: {
        Button("Open Settings", action: actions.openSettings)
          .buttonStyle(.glassProminent)
      }
    }
  }

  // MARK: nothing prepared

  private var noModel: some View {
    card {
      ContentUnavailableView {
        Label("No Model", systemImage: "shippingbox")
      } description: {
        Text(state.catalogUnavailable ? "Couldn't load models. Check your connection." : "Get a model before you drive.")
      } actions: {
        if state.catalogUnavailable {
          Button("Try Again", action: actions.refreshCatalog)
            .buttonStyle(.glass)
        } else {
          if let row = state.defaultModel, ModelStore.canUse(row) {
            Button("Get \(row.displayName)", action: actions.useDefault)
              .buttonStyle(.glassProminent)
          }
          Button("Browse Models", action: actions.openModels)
        }
      }
    }
  }

  // MARK: failed

  private func failed(_ detail: String) -> some View {
    card {
      ContentUnavailableView {
        Label(state.runState == .serving ? "Model Failed" : "Server Stopped", systemImage: "exclamationmark.triangle")
      } description: {
        Text(detail)
          .lineLimit(4)
      } actions: {
        Button("Try Again", action: actions.retry)
          .buttonStyle(.glass)
      }
    }
  }

  private func card(@ViewBuilder _ content: () -> some View) -> some View {
    content()
      .frame(maxWidth: .infinity)
      .padding(.vertical, compact ? 0 : 12)
      .background(Color.cardBackground, in: .rect(cornerRadius: cardCornerRadius, style: .continuous))
  }
}
