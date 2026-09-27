import JetlinkKit
import JetlinkServer
import JetlinkUI
import SwiftUI

/// Connection, performance, display, storage and versions, as a native form.
struct SettingsScreen: View {
  @Environment(AppModel.self) private var app
  @State private var portText = ""

  var body: some View {
    @Bindable var settings = app.settings
    NavigationStack {
      Form {
        connection
        Section {
          Picker("Compute", selection: $settings.device) {
            Text(CoreMLBackend.Device.aneWhole.title).tag(CoreMLBackend.Device.aneWhole)
            Text(CoreMLBackend.Device.ane.title).tag(CoreMLBackend.Device.ane)
            Text(CoreMLBackend.Device.coreml.title).tag(CoreMLBackend.Device.coreml)
            #if targetEnvironment(simulator)
              // The simulator has no Neural Engine and runs CoreML on the CPU anyway.
              Text(CoreMLBackend.Device.cpu.title).tag(CoreMLBackend.Device.cpu)
            #endif
          }
          Toggle("GPU Keep-Alive", isOn: $settings.keepGPUAwake)
        } header: {
          Text("Performance")
        } footer: {
          Text("Changing Compute prepares models again. Keep-Alive holds the GPU at speed between frames.")
        }
        Section {
          Toggle("Keep Screen On", isOn: $settings.keepScreenOn)
        } header: {
          Text("Display")
        } footer: {
          Text("Jetlink has to stay open while you drive.")
        }
        storage
        about
      }
      .navigationTitle("Settings")
      .onAppear { portText = String(settings.port) }
      .onChange(of: settings.device) { app.server.restart() }
      .onChange(of: settings.keepGPUAwake) { app.server.restart() }
    }
  }

  // MARK: connection

  private var connection: some View {
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
      if app.network.addresses.isEmpty {
        LabeledContent("Address", value: "Not Connected")
      }
      ForEach(app.network.addresses) { address in
        LabeledContent {
          Text("\(address.address):\(app.server.port.map(String.init) ?? portText)")
            .monospacedDigit()
            .textSelection(.enabled)
        } label: {
          Label(address.kind.title, systemImage: address.kind.symbol)
        }
      }
    } header: {
      Text("Connection")
    } footer: {
      Text("Set the comma's endpoint to the Ethernet address.")
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
        LabeledContent("Models", value: ByteCount.string(disk.modelsBytes))
        LabeledContent("Engines", value: ByteCount.string(disk.enginesBytes))
        LabeledContent("Available", value: ByteCount.string(disk.freeBytes))
      }
    }
  }

  // MARK: about

  private var about: some View {
    Section("About") {
      LabeledContent("Version", value: Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "")
      LabeledContent("Runtime", value: "onnxruntime \(app.server.info?.runtimeVersion ?? OrtRuntime.version)")
      if let device = app.server.info?.device {
        LabeledContent("Chip", value: SettingsScreen.chip(device))
      }
      LabeledContent("Server", value: serverText)
    }
  }

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
