import JetlinkKit
import JetlinkUI
import SwiftUI
import UIKit

/// The app's one screen: the dashboard, with Models and Settings a tap away.
struct DashboardScreen: View {
  @Environment(AppModel.self) private var app
  @Environment(\.verticalSizeClass) private var verticalSizeClass
  @State private var showingModels = false
  @State private var showingSettings = false

  var body: some View {
    NavigationStack {
      DashboardContent(
        state: app.dashboard,
        landscape: verticalSizeClass == .compact,
        onUseDefault: useDefault,
        onOpenModels: { showingModels = true },
        onRetry: retry,
        onOpenSettings: openSettings
      )
      .navigationTitle("Jetlink")
      .navigationBarTitleDisplayMode(.inline)
      .toolbar {
        ToolbarItem(placement: .topBarLeading) {
          Button("Models", systemImage: "shippingbox") { showingModels = true }
        }
        ToolbarItem(placement: .topBarTrailing) {
          Button("Settings", systemImage: "gearshape") { showingSettings = true }
        }
      }
    }
    .sheet(isPresented: $showingModels) {
      ModelsScreen()
    }
    .sheet(isPresented: $showingSettings) {
      SettingsScreen()
    }
    .alert(
      "Jetlink could not do that",
      isPresented: Binding(get: { app.models.lastError != nil }, set: { if !$0 { app.models.clearError() } })
    ) {
      Button("OK", role: .cancel) { app.models.clearError() }
    } message: {
      Text(app.models.lastError ?? "")
    }
  }

  private func useDefault() {
    guard let row = app.models.rows.first(where: \.isDefault) else { return }
    app.models.use(row, confirmedInterruption: true)
  }

  private func openSettings() {
    if let url = URL(string: UIApplication.openSettingsURLString) {
      UIApplication.shared.open(url)
    }
  }

  /// A failed server starts again; a failed model is asked for again.
  private func retry() {
    if case .failed = app.server.runState {
      app.server.restart()
    } else if let sha = app.server.engine.sha256, let row = app.models.rows.first(where: { $0.sha256 == sha }) {
      app.models.use(row, confirmedInterruption: true)
    } else {
      showingModels = true
    }
  }
}
