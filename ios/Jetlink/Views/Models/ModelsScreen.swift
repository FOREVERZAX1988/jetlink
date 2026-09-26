import JetlinkKit
import JetlinkUI
import SwiftUI
import UniformTypeIdentifiers

/// The catalog and what is on this iPhone: one Use button per model, which
/// downloads, prepares and loads it in one step, and swipes to delete.
struct ModelsScreen: View {
  @Environment(AppModel.self) private var app
  @Environment(\.dismiss) private var dismiss
  @State private var importing = false
  @State private var confirmation: Confirmation?
  @State private var importError: String?
  /// Copies of picked files, removed once their import has finished.
  @State private var stagedImports: Set<String> = []

  enum Confirmation: Identifiable {
    case deleteDownload(ModelRow), deleteEngines(ModelRow), switchModel(ModelRow)

    var row: ModelRow {
      switch self {
      case let .deleteDownload(row), let .deleteEngines(row), let .switchModel(row): row
      }
    }

    var id: String {
      switch self {
      case .deleteDownload: "download-\(row.id)"
      case .deleteEngines: "engines-\(row.id)"
      case .switchModel: "use-\(row.id)"
      }
    }
  }

  private var models: ModelStore { app.models }

  var body: some View {
    NavigationStack {
      content
        .navigationTitle("Models")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
          ToolbarItem(placement: .topBarLeading) {
            Button("Add Model File", systemImage: "plus") { importing = true }
          }
          ToolbarItem(placement: .confirmationAction) {
            Button("Done") { dismiss() }
          }
        }
    }
    .fileImporter(isPresented: $importing, allowedContentTypes: [ModelsScreen.onnxType]) { result in
      switch result {
      case let .success(url): stageImport(url)
      case let .failure(error): importError = error.localizedDescription
      }
    }
    .onChange(of: models.imports) { _, imports in
      for event in imports where ["done", "failed"].contains(event.state) && stagedImports.contains(event.path) {
        try? FileManager.default.removeItem(atPath: event.path)
        stagedImports.remove(event.path)
      }
    }
    .alert(Text(confirmationTitle), isPresented: confirmationPresented, presenting: confirmation) { item in
      Button(confirmButtonTitle(item), role: isDestructive(item) ? .destructive : nil) { perform(item) }
      Button("Cancel", role: .cancel) {}
    } message: { item in
      Text(confirmationMessage(item))
    }
    .alert("Couldn't add that file", isPresented: Binding(get: { importError != nil }, set: { if !$0 { importError = nil } })) {
      Button("OK") { importError = nil }
    } message: {
      Text(importError ?? "")
    }
  }

  @ViewBuilder
  private var content: some View {
    if models.catalog == nil && models.rows.isEmpty {
      ProgressView("Loading the model list…")
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    } else {
      List {
        if let error = models.catalog?.error, !error.isEmpty {
          Section {
            Label(models.catalog?.models.isEmpty ?? true ? "Model list unavailable" : "Model list not refreshed", systemImage: "exclamationmark.triangle.fill")
              .foregroundStyle(.orange)
            Text(error)
              .font(.footnote)
              .foregroundStyle(.secondary)
          }
        }
        Section {
          rows(models.rows.filter { !$0.isLocal && !$0.isOrphan })
        } header: {
          Text("sunnypilot Models")
        } footer: {
          Text("Use a model to download it, prepare it for this iPhone and serve it to the comma. Only the download needs a network connection; keep Jetlink open until it finishes.")
        }
        let local = models.rows.filter { $0.isLocal && !$0.isOrphan }
        if !local.isEmpty {
          Section("Added to This iPhone") { rows(local) }
        }
        let orphans = models.rows.filter(\.isOrphan)
        if !orphans.isEmpty {
          Section("Unrecognized Files") { rows(orphans) }
        }
        if let disk = models.inventory?.disk {
          Section {
          } footer: {
            Text(ModelFormatting.diskSummary(models: disk.modelsBytes, engines: disk.enginesBytes, free: disk.freeBytes))
          }
        }
      }
      .refreshable { models.refreshCatalog() }
    }
  }

  private func rows(_ rows: [ModelRow]) -> some View {
    ForEach(rows) { row in
      ModelRowView(
        row: row,
        isCheckingCatalog: models.catalog == nil,
        use: { startUse(row) },
        cancel: { models.cancelDownload(row) },
        stop: { models.unload() })
        .swipeActions(edge: .trailing, allowsFullSwipe: false) {
          if !row.preparedFor.isEmpty {
            Button("Delete Engines", systemImage: "trash", role: .destructive) { confirmation = .deleteEngines(row) }
          }
          if hasModelFile(row) {
            Button("Delete Download", systemImage: "arrow.down.circle.dotted") { confirmation = .deleteDownload(row) }
              .tint(.orange)
          }
        }
        .contextMenu { menu(for: row) }
    }
  }

  @ViewBuilder
  private func menu(for row: ModelRow) -> some View {
    if ModelStore.canUse(row) {
      Button("Use Model", systemImage: "play.circle") { startUse(row) }
    }
    if case .downloading = row.status {
      Button("Cancel Download", systemImage: "xmark.circle") { models.cancelDownload(row) }
    }
    if row.status == .loaded {
      Button("Stop Using Model", systemImage: "stop.circle") { models.unload() }
    }
    if hasModelFile(row) {
      Button("Delete Download…", systemImage: "arrow.down.circle.dotted", role: .destructive) { confirmation = .deleteDownload(row) }
    }
    if !row.preparedFor.isEmpty {
      Button("Delete Prepared Engines…", systemImage: "trash", role: .destructive) { confirmation = .deleteEngines(row) }
    }
  }

  // MARK: actions

  private func startUse(_ row: ModelRow) {
    if models.useNeedsConfirmation(row) {
      confirmation = .switchModel(row)
    } else {
      models.use(row)
    }
  }

  private func hasModelFile(_ row: ModelRow) -> Bool {
    guard let sha = row.sha256 else { return false }
    return models.inventory?.models.contains { $0.sha256 == sha } ?? false
  }

  /// A picked file is only readable while its security scope is open, and
  /// the import runs later on the server's own time, so it gets a copy of its own.
  private func stageImport(_ url: URL) {
    let scoped = url.startAccessingSecurityScopedResource()
    defer { if scoped { url.stopAccessingSecurityScopedResource() } }
    let staged = URL.temporaryDirectory.appending(path: "import-\(UUID().uuidString.prefix(8))-\(url.lastPathComponent)")
    do {
      try FileManager.default.copyItem(at: url, to: staged)
    } catch {
      importError = error.localizedDescription
      return
    }
    stagedImports.insert(staged.path)
    models.importModel(at: staged)
  }

  private func perform(_ item: Confirmation) {
    switch item {
    case let .deleteDownload(row): models.forget(row, artifacts: false, model: true)
    case let .deleteEngines(row): models.forget(row, artifacts: true, model: false)
    case let .switchModel(row): models.use(row, confirmedInterruption: true)
    }
  }

  // MARK: alerts

  private var confirmationPresented: Binding<Bool> {
    Binding(get: { confirmation != nil }, set: { if !$0 { confirmation = nil } })
  }

  private var confirmationTitle: String {
    switch confirmation {
    case .deleteDownload: "Delete the download?"
    case .deleteEngines: "Delete the prepared engines?"
    case let .switchModel(row): "Use \(row.displayName)?"
    case nil: ""
    }
  }

  private func confirmButtonTitle(_ item: Confirmation) -> String {
    switch item {
    case .deleteDownload, .deleteEngines: "Delete"
    case .switchModel: "Use Model"
    }
  }

  private func isDestructive(_ item: Confirmation) -> Bool {
    switch item {
    case .deleteDownload, .deleteEngines: true
    case .switchModel: false
    }
  }

  private func confirmationMessage(_ item: Confirmation) -> String {
    switch item {
    case let .deleteDownload(row):
      let size = row.bytes.map { "\(ByteCount.string($0)) " } ?? ""
      return "Deletes the \(size)model file for \(row.displayName). The prepared engine stays, so the comma can still use this model."
    case let .deleteEngines(row):
      let bytes = row.preparedFor.reduce(Int64(0)) { $0 + $1.bytes }
      var text = "Deletes every prepared engine for \(row.displayName), \(ByteCount.string(bytes)) in all. Using it again prepares it again."
      if row.isLoaded {
        text += " Jetlink stops using it first."
      }
      return text
    case let .switchModel(row):
      let current = models.rows.first { $0.isLoaded }?.displayName ?? "another model"
      return "The comma is using \(current). Switching drops it to its small model until \(row.displayName) is ready."
    }
  }

  static let onnxType = UTType(filenameExtension: "onnx") ?? .data
}

/// One model: its name and tags, one line of facts, and on the trailing edge
/// whatever applies now: Use, progress, or In Use.
struct ModelRowView: View {
  let row: ModelRow
  let isCheckingCatalog: Bool
  let use: () -> Void
  let cancel: () -> Void
  let stop: () -> Void

  var body: some View {
    HStack(spacing: 12) {
      VStack(alignment: .leading, spacing: 3) {
        HStack(spacing: 6) {
          Text(row.displayName)
            .font(.body.weight(.medium))
            .lineLimit(1)
          if row.isDefault { ModelTag("Default", tone: .accentColor) }
          if row.isRequestedByComma { ModelTag("Comma", tone: .green) }
        }
        let detail = ModelFormatting.detailLine(row)
        if !detail.isEmpty {
          Text(detail)
            .font(.caption)
            .foregroundStyle(.secondary)
            .lineLimit(1)
        }
        status
      }
      Spacer(minLength: 8)
      accessory
    }
    .padding(.vertical, 2)
  }

  @ViewBuilder
  private var status: some View {
    switch row.status {
    case let .downloading(frac, rateBps):
      ProgressView(value: min(max(frac, 0), 1)) {
        Text(ModelFormatting.downloadCaption(frac: frac, rateBps: rateBps))
          .font(.caption)
          .foregroundStyle(.secondary)
          .monospacedDigit()
      }
      .padding(.top, 2)
    case let .preparing(stage, frac, msg):
      ProgressView(value: min(max(frac, 0), 1)) {
        Text(msg.isEmpty ? ProgressRow.stageName(stage) : msg)
          .font(.caption)
          .foregroundStyle(.secondary)
          .lineLimit(1)
      }
      .padding(.top, 2)
    case let .failed(detail):
      Text(detail)
        .font(.caption)
        .foregroundStyle(.red)
        .lineLimit(2)
    case .unresolved:
      Text(isCheckingCatalog ? "Checking…" : "Size unknown")
        .font(.caption)
        .foregroundStyle(.secondary)
    default:
      EmptyView()
    }
  }

  @ViewBuilder
  private var accessory: some View {
    switch row.status {
    case .loaded:
      Menu {
        Button("Stop Using Model", systemImage: "stop.circle", action: stop)
      } label: {
        Label("In Use", systemImage: "checkmark.circle.fill")
          .font(.subheadline.weight(.semibold))
          .foregroundStyle(.green)
      }
    case .downloading:
      Button("Cancel", systemImage: "xmark.circle.fill", action: cancel)
        .labelStyle(.iconOnly)
        .font(.title3)
        .foregroundStyle(.secondary)
        .buttonStyle(.plain)
    case .preparing:
      ProgressView()
    case .unresolved:
      EmptyView()
    case .notDownloaded, .downloaded, .prepared, .failed:
      if ModelStore.canUse(row) {
        Button(row.status == .failed("") ? "Retry" : "Use", action: use)
          .buttonStyle(.bordered)
          .buttonBorderShape(.capsule)
          .controlSize(.small)
          .accessibilityHint(ModelFormatting.useHelp(row, device: "this iPhone"))
      }
    }
  }
}
