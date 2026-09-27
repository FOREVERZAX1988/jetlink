import JetlinkKit
import JetlinkUI
import SwiftUI

/// What the server and the app have logged, newest at the bottom, following
/// new lines until the reader scrolls up. Warnings in orange, errors in red.
struct LogsScreen: View {
  @Environment(AppModel.self) private var app
  @State private var follow = true

  private static let bottomAnchor = "logs.bottom"

  var body: some View {
    let logs = app.server.logs
    ScrollViewReader { proxy in
      ScrollView {
        LazyVStack(alignment: .leading, spacing: 2) {
          if logs.lines.isEmpty {
            Text("Nothing logged yet.")
              .foregroundStyle(.secondary)
              .frame(maxWidth: .infinity, alignment: .center)
              .padding(.top, 40)
          }
          ForEach(Array(logs.lines.enumerated()), id: \.offset) { _, line in
            Text(line)
              .font(.caption.monospaced())
              .foregroundStyle(LogsScreen.tone(for: line))
              .textSelection(.enabled)
              .frame(maxWidth: .infinity, alignment: .leading)
          }
          Color.clear
            .frame(height: 1)
            .id(LogsScreen.bottomAnchor)
        }
        .padding(.horizontal, StatusContent.margin)
        .padding(.vertical, 8)
      }
      .onScrollGeometryChange(for: Bool.self) { geometry in
        geometry.contentOffset.y + geometry.containerSize.height >= geometry.contentSize.height - 40
      } action: { _, atBottom in
        follow = atBottom
      }
      .onChange(of: logs.revision) {
        if follow {
          proxy.scrollTo(LogsScreen.bottomAnchor, anchor: .bottom)
        }
      }
      .onAppear {
        proxy.scrollTo(LogsScreen.bottomAnchor, anchor: .bottom)
      }
    }
    .background(Color.groupedBackground)
    .navigationTitle("Logs")
    .navigationBarTitleDisplayMode(.inline)
    .toolbar {
      ToolbarItem(placement: .topBarTrailing) {
        ShareLink(item: logs.lines.joined(separator: "\n"), preview: SharePreview("Jetlink logs"))
          .disabled(logs.lines.isEmpty)
      }
      ToolbarItem(placement: .topBarTrailing) {
        Button("Clear", systemImage: "trash") { logs.clear() }
          .disabled(logs.lines.isEmpty)
      }
    }
  }

  /// Lines read "time LEVEL category: message".
  static func tone(for line: String) -> Color {
    LogTone.color(for: line)
  }
}
