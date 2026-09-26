import SwiftUI
import UIKit

@main
struct JetlinkPhoneApp: App {
  @State private var app = AppModel()
  @Environment(\.scenePhase) private var scenePhase

  var body: some Scene {
    WindowGroup {
      RootView()
        .environment(app)
        .onAppear { app.launch() }
        .onChange(of: scenePhase) { _, phase in
          if phase == .active {
            app.server.becameActive()
            app.device.refresh()
            app.network.refresh()
            app.localNetwork.check()
            app.retryCatalogIfEmpty()
          }
          updateIdleTimer()
        }
        .onChange(of: app.settings.keepScreenOn) { updateIdleTimer() }
        .onChange(of: app.network.addresses) { app.retryCatalogIfEmpty() }
        .onChange(of: app.server.runState) { updateIdleTimer() }
    }
  }

  /// The screen stays on while the server runs and Jetlink is in front: a
  /// locked phone suspends the app, and the comma loses the big model.
  private func updateIdleTimer() {
    UIApplication.shared.isIdleTimerDisabled =
      app.settings.keepScreenOn && app.server.runState == .serving && scenePhase == .active
  }
}
