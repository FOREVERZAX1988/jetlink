import JetlinkKit
import JetlinkORT
import JetlinkUI
import SwiftUI

/// Connection, performance, display, storage, help and versions, as a native form.
struct SettingsScreen: View {
  @Environment(AppModel.self) private var app
  @State private var portText = ""
  @State private var versionTaps = 0
  @State private var path: [Destination] = SettingsScreen.initialPath

  enum Destination: Hashable {
    case logs, connect
  }

  /// `-tab logs` or `-tab connect` opens Settings with that screen pushed,
  /// for screenshots.
  static var initialPath: [Destination] {
    switch UserDefaults.standard.string(forKey: "tab") {
    case "logs": [.logs]
    case "connect": [.connect]
    default: []
    }
  }

  var body: some View {
    @Bindable var settings = app.settings
    NavigationStack(path: $path) {
      Form {
        connection
        Section {
          Picker("Processor", selection: $settings.device) {
            Text(OrtProfile.ane.title).tag(OrtProfile.ane)
            Text(OrtProfile.coreml.title).tag(OrtProfile.coreml)
            #if targetEnvironment(simulator)
              // The simulator has no Neural Engine and runs CoreML on the CPU anyway.
              Text(OrtProfile.cpu.title).tag(OrtProfile.cpu)
            #endif
          }
          Toggle("Keep CPU Awake", isOn: $settings.keepCPUWarm)
          Toggle("Keep GPU Awake", isOn: $settings.keepGPUAwake)
        } header: {
          Text("Performance")
        } footer: {
          Text("Changing the processor prepares models again.")
        }
        Section {
          Picker("Appearance", selection: $settings.appearance) {
            Text("System").tag(AppearanceChoice.system)
            Text("Light").tag(AppearanceChoice.light)
            Text("Dark").tag(AppearanceChoice.dark)
          }
          Toggle("Keep Screen On", isOn: $settings.keepScreenOn)
        } header: {
          Text("Display")
        } footer: {
          Text("Jetlink has to stay open while you drive.")
        }
        storage
        help
        about
        if settings.developer {
          developer
        }
      }
      .readableWidth()
      .navigationTitle("Settings")
      .navigationDestination(for: Destination.self) { destination in
        switch destination {
        case .logs: LogsScreen()
        case .connect: ConnectHelpScreen()
        }
      }
      .onAppear { portText = String(settings.port) }
      .onChange(of: settings.device) { app.server.restart() }
      .onChange(of: settings.keepGPUAwake) { app.server.restart() }
      .onChange(of: settings.keepCPUWarm) { app.server.restart() }
      .onChange(of: settings.developer) { app.server.restart() }
      .onChange(of: settings.wifiLink) {
        app.server.restart()
        app.syncPresence()
      }
      .sensoryFeedback(.success, trigger: settings.developer)
    }
  }

  // MARK: connection

  /// The Link row: a live link's medium, a cable or a comma on this phone's
  /// hotspot on its way, and nothing otherwise. A comma that answered on
  /// another hotspot is not this phone's, so it is not "Connecting".
  private var linkValue: String {
    if let medium = app.server.linkMedium { return medium.phoneTitle }
    if app.network.cable != nil { return "Connecting" }
    if app.presence.state == .here { return "Connecting" }
    return "Not Connected"
  }

  private var connection: some View {
    @Bindable var settings = app.settings
    return Section {
      LabeledContent("Link", value: linkValue)
      Toggle("Wi-Fi Link", isOn: $settings.wifiLink)
    } header: {
      Text("Connection")
    } footer: {
      if settings.wifiLink {
        Text(
          "For a comma set to Wi-Fi. Turn on Personal Hotspot (Allow Others to Join on, Maximize Compatibility off) and join the comma to it."
        )
      }
    }
  }

  // MARK: developer

  /// The port bench tools on a Mac reach the phone on over Wi-Fi. Driving
  /// never uses it: over the cable the phone dials the comma.
  private var developer: some View {
    Section {
      LabeledContent("Port") {
        TextField("5599", text: $portText)
          .keyboardType(.numberPad)
          .multilineTextAlignment(.trailing)
          .monospacedDigit()
          .onSubmit(applyPort)
      }
      if portChanged {
        Button("Use Port \(portText)", action: applyPort)
      }
      // The cable's address is the comma's to hand out and the phone's to
      // dial; nobody types it.
      ForEach(app.network.addresses.filter { $0.kind != .cable }) { address in
        LabeledContent {
          Text("\(address.address):\(app.server.port.map(String.init) ?? portText)")
            .monospacedDigit()
            .textSelection(.enabled)
        } label: {
          Label(address.kind.title, systemImage: address.kind.symbol)
        }
      }
    } header: {
      Text("Developer")
    } footer: {
      Text("Lets bench tools on a Mac reach the phone over Wi-Fi. Driving does not need it.")
    }
  }

  private var portChanged: Bool {
    UInt16(portText).map { $0 != app.settings.port } ?? false
  }

  private func applyPort() {
    guard let port = UInt16(portText), port > 0 else {
      portText = String(app.settings.port)
      return
    }
    guard port != app.settings.port else { return }
    app.settings.port = port
    app.server.restart()
  }

  // MARK: storage

  @ViewBuilder
  private var storage: some View {
    if let disk = app.models.inventory?.disk {
      Section("Storage") {
        LabeledContent("Downloaded", value: ByteCount.string(disk.modelsBytes))
        LabeledContent("Prepared", value: ByteCount.string(disk.enginesBytes))
        LabeledContent("Available", value: ByteCount.string(disk.freeBytes))
      }
    }
  }

  // MARK: help

  private var help: some View {
    Section("Help") {
      NavigationLink("Connecting the Comma", value: Destination.connect)
      NavigationLink("Logs", value: Destination.logs)
    }
  }

  // MARK: about

  private var about: some View {
    Section("About") {
      LabeledContent("Version", value: Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "")
        .contentShape(Rectangle())
        .onTapGesture(perform: tapVersion)
      LabeledContent("Runtime", value: "onnxruntime \(app.server.state.server?.runtimeVersion ?? OrtRuntime.version)")
      if let device = app.server.state.server?.device {
        LabeledContent("Chip", value: SettingsScreen.chip(device))
      }
      LabeledContent("Server", value: serverText)
    }
  }

  /// Seven taps on the version turn the developer settings on or off, as a
  /// build number does on Android.
  private func tapVersion() {
    versionTaps += 1
    guard versionTaps >= SettingsScreen.developerTaps else { return }
    versionTaps = 0
    app.settings.developer.toggle()
  }

  static let developerTaps = 7

  /// "Apple A17 Pro" from "ane-Apple_A17_Pro".
  static func chip(_ device: String) -> String {
    let name = device.split(separator: "-", maxSplits: 1).last.map(String.init) ?? device
    return name.replacingOccurrences(of: "_", with: " ")
  }

  private var serverText: String {
    switch app.server.runState {
    case .stopped: "Stopped"
    case .starting: "Starting"
    case .serving: "Running"
    case .stopping: "Stopping"
    case .failed: "Failed"
    }
  }
}
