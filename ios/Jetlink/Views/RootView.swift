import JetlinkKit
import JetlinkUI
import SwiftUI

/// Three tabs: Status, Models and Settings. Away from Status, the state rides
/// along in a pill above the tab bar, the way Music keeps what is playing in
/// view, and a tap on it goes back.
struct RootView: View {
  enum Tab: Hashable {
    case status, models, settings
  }

  @Environment(AppModel.self) private var app
  @State private var tab: Tab = RootView.initialTab

  var body: some View {
    TabView(selection: $tab) {
      SwiftUI.Tab("Status", systemImage: "gauge.with.needle.fill", value: .status) {
        StatusScreen(tab: $tab)
      }
      SwiftUI.Tab("Models", systemImage: "shippingbox.fill", value: .models) {
        ModelsScreen()
      }
      SwiftUI.Tab("Settings", systemImage: "gearshape.fill", value: .settings) {
        SettingsScreen()
      }
    }
    .tabBarMinimizeBehavior(.onScrollDown)
    .tabViewBottomAccessory(isEnabled: tab != .status) {
      StatusAccessory(state: app.status)
        .onTapGesture { tab = .status }
    }
  }

  /// `-tab models` or `-tab settings` on the command line opens there, for
  /// screenshots from the simulator.
  static var initialTab: Tab {
    switch UserDefaults.standard.string(forKey: "tab") {
    case "models": .models
    case "settings": .settings
    default: .status
    }
  }
}

/// The state in one line: the summary's symbol and word, then the headroom
/// while serving, or the model.
struct StatusAccessory: View {
  let state: StatusState
  @Environment(\.tabViewBottomAccessoryPlacement) private var placement

  var body: some View {
    let summary = state.summary
    HStack(spacing: 10) {
      Image(systemName: summary.symbol)
        .foregroundStyle(summary.tone.color)
        .font(.body.weight(.semibold))
      Text(summary.title)
        .font(.subheadline.weight(.semibold))
      Spacer(minLength: 8)
      Text(detail)
        .font(.subheadline.monospacedDigit())
        .foregroundStyle(.secondary)
        .lineLimit(1)
    }
    .padding(.horizontal, placement == .inline ? 12 : 16)
    .contentShape(.rect)
    .accessibilityElement(children: .combine)
  }

  private var detail: String {
    if let recent = state.recent, state.isServingFrames {
      return FrameBudgetView.headroomText(p99: recent.served.p99)
    }
    return state.modelName ?? ""
  }
}
