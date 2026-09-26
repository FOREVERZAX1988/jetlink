import JetlinkKit
import JetlinkServer
import JetlinkUI
import SwiftUI

/// The connection, where the model runs, the screen, storage, and versions.
struct SettingsScreen: View {
  @Environment(AppModel.self) private var app
  @Environment(\.dismiss) private var dismiss
  @State private var portText = ""
  @State private var confirmingRestart = false

  var body: some View {
    @Bindable var settings = app.settings
    NavigationStack {
      Form {
        connection
        Section {
          Picker("Run the Model On", selection: $settings.device) {
            Text(CoreMLBackend.Device.ane.title).tag(CoreMLBackend.Device.ane)
            Text(CoreMLBackend.Device.coreml.title).tag(CoreMLBackend.Device.coreml)
          }
          Toggle("Keep the GPU Clocked Up", isOn: $settings.keepGPUAwake)
        } header: {
          Text("Model")
        } footer: {
          Text(
            "The vision half of the model runs on the Neural Engine and the rest on the GPU, the fastest layout on a Mac. GPU only is for when another app keeps the Neural Engine busy; each choice prepares the model again. A small GPU job between frames stops the GPU slowing down in the gaps, at some cost in power."
          )
        }
        Section {
          Toggle("Keep the Screen On", isOn: $settings.keepScreenOn)
        } header: {
          Text("While Jetlink Is Open")
        } footer: {
          Text(
            "iOS suspends Jetlink when the iPhone locks or another app comes to the front, and the comma then drives on its small model. Keep Jetlink on screen while driving, and keep the iPhone on power."
          )
        }
        storage
        about
      }
      .navigationTitle("Settings")
      .navigationBarTitleDisplayMode(.inline)
      .toolbar {
        ToolbarItem(placement: .confirmationAction) {
          Button("Done") { dismiss() }
        }
      }
      .onAppear { portText = String(settings.port) }
      .onChange(of: settings.device) { app.server.restart() }
      .onChange(of: settings.keepGPUAwake) { app.server.restart() }
    }
  }

  // MARK: connection

  @ViewBuilder
  private var connection: some View {
    Section {
      LabeledContent("Port") {
        TextField("5599", text: $portText)
          .keyboardType(.numberPad)
          .multilineTextAlignment(.trailing)
          .onSubmit(applyPort)
      }
      if portChanged {
        Button("Listen on Port \(portText)", action: applyPort)
      }
      if app.network.addresses.isEmpty {
        Text("No network. Connect a USB-C Ethernet adapter.")
          .foregroundStyle(.secondary)
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
      Text("Comma Connection")
    } footer: {
      Text("Set the comma's JetlinkEndpoint to the Ethernet address. Wired Ethernet through a USB-C adapter meets the 50 ms budget; Wi-Fi is for testing.")
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
        LabeledContent("Downloaded Models", value: ByteCount.string(disk.modelsBytes))
        LabeledContent("Prepared Engines", value: ByteCount.string(disk.enginesBytes))
        LabeledContent("Available", value: ByteCount.string(disk.freeBytes))
      }
    }
  }

  // MARK: about

  private var about: some View {
    Section("About") {
      LabeledContent("Version", value: Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "")
      LabeledContent("onnxruntime", value: app.server.info?.runtimeVersion ?? OrtRuntime.version)
      if let device = app.server.info?.device {
        LabeledContent("Engine", value: device.replacingOccurrences(of: "_", with: " "))
      }
      LabeledContent("Server") {
        StatusBadge(text: serverText, tone: serverTone)
      }
    }
  }

  private var serverText: String {
    switch app.server.runState {
    case .stopped: "Stopped"
    case .starting: "Starting…"
    case .serving: "Serving"
    case .stopping: "Stopping…"
    case .failed: "Failed"
    }
  }

  private var serverTone: StatusBadge.Tone {
    switch app.server.runState {
    case .serving: .good
    case .starting: .info
    case .failed: .bad
    case .stopped, .stopping: .neutral
    }
  }
}
