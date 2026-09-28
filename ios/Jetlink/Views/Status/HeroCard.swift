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
    case .failed:
      failed
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
          Figure("P99", ms: stats?.servedMs.p99)
          Divider().frame(height: 32)
          Figure("Max", ms: stats?.servedMs.max)
        }
      }
    }
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
        Text(state.cableAddress == nil ? "Plug in the comma." : "Connecting over USB.")
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

  /// What went wrong, in a line; the error itself is in Logs.
  private var failed: some View {
    let model = state.runState == .serving
    return card {
      ContentUnavailableView {
        Label(model ? "Model Failed" : "Jetlink Stopped", systemImage: "exclamationmark.triangle")
      } description: {
        Text(model ? "Couldn't prepare this model. See Logs for details." : "See Logs for details.")
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
