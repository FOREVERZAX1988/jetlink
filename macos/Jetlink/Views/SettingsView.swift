import AppKit
import JetlinkUI
import SwiftUI

struct SettingsView: View {
  var body: some View {
    TabView {
      GeneralSettingsView()
        .tabItem { Label("General", systemImage: "gear") }
      ServerSettingsView()
        .tabItem { Label("Server", systemImage: "cpu") }
    }
    .frame(width: 540)
  }
}

// MARK: - General

struct GeneralSettingsView: View {
  @Environment(AppSettings.self) private var settings
  @Environment(ServerStore.self) private var server
  @State private var loginItem = LoginItem()

  var body: some View {
    @Bindable var settings = settings
    Form {
      Section {
        Toggle("Start server when Jetlink opens", isOn: $settings.startServerOnLaunch)
        VStack(alignment: .leading, spacing: 4) {
          Toggle("Open Jetlink at login", isOn: $loginItem.isEnabled)
          if loginItem.requiresApproval {
            Text("Approve Jetlink in System Settings > General > Login Items.")
              .font(.callout)
              .foregroundStyle(.secondary)
            Button("Open Login Items") { loginItem.openSystemSettings() }
          }
        }
        VStack(alignment: .leading, spacing: 4) {
          Toggle("Keep the Mac awake while serving", isOn: $settings.keepAwakeWhileServing)
          Text("Only when connected to power. On battery, keep the lid open.")
            .font(.callout)
            .foregroundStyle(.secondary)
        }
      }

      Section("Cache Folder") {
        VStack(alignment: .leading, spacing: 8) {
          Text(settings.cacheDirectory.path(percentEncoded: false))
            .font(.system(.callout, design: .monospaced))
            .textSelection(.enabled)
            .fixedSize(horizontal: false, vertical: true)
          HStack {
            Button("Choose…") { chooseCacheDirectory() }
            Button("Show in Finder") {
              NSWorkspace.shared.activateFileViewerSelecting([settings.cacheDirectory])
            }
            if needsRestart {
              Button("Restart Now") { server.restart() }
            }
          }
          Text("Models and prepared engines. A CoreML engine is about 2 GB. Takes effect when the server restarts.")
            .font(.callout)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
        }
      }
    }
    .formStyle(.grouped)
  }

  private var needsRestart: Bool {
    guard server.runState == .serving, let running = server.info?.cache else { return false }
    return running != settings.cacheDirectory.path(percentEncoded: false)
  }

  private func chooseCacheDirectory() {
    let panel = NSOpenPanel()
    panel.canChooseDirectories = true
    panel.canChooseFiles = false
    panel.allowsMultipleSelection = false
    panel.canCreateDirectories = true
    panel.directoryURL = settings.cacheDirectory
    panel.prompt = "Choose"
    if panel.runModal() == .OK, let url = panel.url {
      settings.cacheDirectory = url
    }
  }
}

// MARK: - Server

struct ServerSettingsView: View {
  @Environment(AppSettings.self) private var settings
  @Environment(ServerStore.self) private var server
  @State private var advancedExpanded = false

  var body: some View {
    @Bindable var settings = settings
    Form {
      Section {
        VStack(alignment: .leading, spacing: 4) {
          Picker("Server", selection: $settings.serverEngine) {
            ForEach(ServerEngine.allCases, id: \.self) { engine in
              Text(engine.title).tag(engine)
            }
          }
          Text(ServerSettingsView.engineCaption(settings.serverEngine))
            .font(.callout)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
        }
        VStack(alignment: .leading, spacing: 4) {
          Picker("Backend", selection: $settings.backend) {
            ForEach(BackendChoice.allCases, id: \.self) { choice in
              Text(choice == .auto ? "Automatic (\(choice.title))" : choice.title).tag(choice)
            }
          }
          Text(ServerSettingsView.backendCaption(settings.backend))
            .font(.callout)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
        }
        Picker("Connection", selection: $settings.transport) {
          Text("USB (the comma)").tag(TransportChoice.usb)
          Text("TCP (bench client)").tag(TransportChoice.tcp)
        }
        if settings.transport == .tcp {
          TextField("Port", value: $settings.tcpPort, format: .number.grouping(.never))
        }
        Picker("Log level", selection: $settings.logLevel) {
          Text("Normal (INFO)").tag("INFO")
          Text("Verbose (DEBUG)").tag("DEBUG")
        }
      } footer: {
        HStack {
          Text("Changes apply when the server restarts.")
            .font(.callout)
            .foregroundStyle(.secondary)
          Spacer()
          Button("Restart Server") { server.restart() }
            .disabled(server.runState != .serving)
        }
      }

      if settings.serverEngine == .python {
        Section {
          DisclosureGroup("Advanced", isExpanded: $advancedExpanded) {
            VStack(alignment: .leading, spacing: 6) {
              TextField(
                "Python interpreter override",
                text: Binding(
                  get: { settings.pythonOverride ?? "" },
                  set: { settings.pythonOverride = $0.isEmpty ? nil : $0 }
                ))
              Text("For development. Empty means the bundled runtime.")
                .font(.callout)
                .foregroundStyle(.secondary)
              Text(bundledText)
                .font(.callout)
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
            }
            .padding(.top, 4)
          }
        }
      }
    }
    .formStyle(.grouped)
  }

  static func engineCaption(_ engine: ServerEngine) -> String {
    switch engine {
    case .python:
      "The server Jetlink has always run, in a Python runtime inside the app. Measured on a comma, and the default for now."
    case .swift:
      "The server the iPhone app runs, built into Jetlink: no Python to find or start, and about 1 ms faster a frame on an M1 Pro. Not yet measured with a comma on USB."
    }
  }

  static func backendCaption(_ backend: BackendChoice) -> String {
    switch backend {
    case .auto:
      "Recommended: the fastest on Apple silicon. Preparing takes about 20 seconds the first time."
    case .coreml:
      "Slower. Use it if another app keeps the Neural Engine busy."
    }
  }

  private var bundledText: String {
    guard let manifest = EmbeddedPython.manifest() else {
      return "This build has no bundled Python runtime."
    }
    var parts: [String] = []
    if let python = manifest["python"] { parts.append("Python \(python)") }
    if let ort = manifest["onnxruntime"] { parts.append("onnxruntime \(ort)") }
    return parts.isEmpty ? "This build has no bundled Python runtime." : "Bundled: " + parts.joined(separator: ", ")
  }
}

#Preview("General") {
  GeneralSettingsView()
    .environment(AppSettings.preview())
    .environment(ServerStore.preview(runState: .serving, info: PreviewData.serverInfo, link: PreviewData.linkWaiting, engine: PreviewData.engineReady))
    .frame(width: 540, height: 420)
}

#Preview("Server") {
  ServerSettingsView()
    .environment(AppSettings.preview())
    .environment(ServerStore.preview(runState: .serving, info: PreviewData.serverInfo, link: PreviewData.linkWaiting, engine: PreviewData.engineReady))
    .frame(width: 540, height: 420)
}
