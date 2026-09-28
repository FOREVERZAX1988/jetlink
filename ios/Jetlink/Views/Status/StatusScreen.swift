import JetlinkKit
import JetlinkUI
import SwiftUI
import UIKit

/// The Status tab: the headroom and everything behind it, under a large title
/// whose subtitle says where things stand.
struct StatusScreen: View {
  @Environment(AppModel.self) private var app
  @Environment(\.verticalSizeClass) private var verticalSizeClass
  @Binding var tab: RootView.Tab

  var body: some View {
    let state = app.status
    let landscape = verticalSizeClass == .compact
    NavigationStack {
      StatusContent(state: state, landscape: landscape, actions: actions)
        .navigationTitle("Jetlink")
        .navigationSubtitle(state.subtitle)
        .toolbarTitleDisplayMode(landscape ? .inline : .large)
        .toolbar {
          ToolbarItem(placement: .topBarTrailing) {
            NavigationLink {
              LogsScreen()
            } label: {
              Label("Logs", systemImage: "doc.text")
            }
          }
        }
    }
    // On its side the phone is a dashboard: nothing but the numbers.
    .toolbarVisibility(landscape ? .hidden : .automatic, for: .tabBar)
    .sensoryFeedback(trigger: state.link.state) { old, new in
      if new == .connected { return .success }
      if old == .connected { return .warning }
      return nil
    }
  }

  private var actions: StatusActions {
    StatusActions(
      useDefault: useDefault,
      openModels: { tab = .models },
      retry: retry,
      openSettings: openSettings,
      refreshCatalog: { app.models.refreshCatalog() })
  }

  private func useDefault() {
    guard let row = app.models.rows.first(where: \.isDefault) else { return }
    app.models.use(row, confirmedInterruption: true)
  }

  /// A stopped server starts again; a failed model is asked for again.
  private func retry() {
    if case .failed = app.server.runState {
      app.server.restart()
    } else if let sha = app.server.engine.sha256, let row = app.models.rows.first(where: { $0.sha256 == sha }) {
      app.models.use(row, confirmedInterruption: true)
    } else {
      tab = .models
    }
  }

  private func openSettings() {
    if let url = URL(string: UIApplication.openSettingsURLString) {
      UIApplication.shared.open(url)
    }
  }
}
