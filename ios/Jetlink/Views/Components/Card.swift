import SwiftUI

#if canImport(UIKit)
import UIKit
#else
import AppKit
#endif

extension Color {
  /// The screen behind the cards: grouped, as Settings and Health draw it.
  static var groupedBackground: Color {
    #if canImport(UIKit)
    Color(uiColor: .systemGroupedBackground)
    #else
    Color(nsColor: NSColor(name: nil) { $0.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua ? .black : NSColor(srgbRed: 0.949, green: 0.949, blue: 0.969, alpha: 1) })
    #endif
  }

  /// A card's surface on the grouped background.
  static var cardBackground: Color {
    #if canImport(UIKit)
    Color(uiColor: .secondarySystemGroupedBackground)
    #else
    Color(nsColor: NSColor(name: nil) { $0.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua ? NSColor(srgbRed: 0.11, green: 0.11, blue: 0.118, alpha: 1) : .white })
    #endif
  }
}

/// A rounded content surface with an optional title, the dashboard's unit.
/// Content, not a control, so no glass: Liquid Glass belongs to the bars and
/// buttons that float above it.
struct Card<Content: View>: View {
  var title: String?
  var systemImage: String?
  @ViewBuilder var content: Content

  var body: some View {
    VStack(alignment: .leading, spacing: 12) {
      if let title {
        Label {
          Text(title)
        } icon: {
          if let systemImage {
            Image(systemName: systemImage)
          }
        }
        .font(.subheadline.weight(.semibold))
        .foregroundStyle(.secondary)
        .labelStyle(.titleAndIcon)
      }
      content
    }
    .padding(16)
    .frame(maxWidth: .infinity, alignment: .leading)
    .background(Color.cardBackground, in: .rect(cornerRadius: 22, style: .continuous))
  }
}

/// One headline number with its label: the dashboard's KPI tiles.
struct StatTile: View {
  let label: String
  let value: String
  var detail: String?
  var systemImage: String?
  var tone: Color?

  var body: some View {
    VStack(alignment: .leading, spacing: 4) {
      HStack(spacing: 5) {
        if let systemImage {
          Image(systemName: systemImage)
            .foregroundStyle(tone ?? .secondary)
        }
        Text(label)
          .foregroundStyle(.secondary)
      }
      .font(.subheadline)
      .lineLimit(1)
      Text(value)
        .font(.title2.weight(.semibold))
        .contentTransition(.numericText())
        .lineLimit(1)
        .minimumScaleFactor(0.6)
      if let detail {
        Text(detail)
          .font(.caption)
          .foregroundStyle(tone ?? .secondary)
          .lineLimit(1)
      }
    }
    .padding(14)
    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    .background(Color.cardBackground, in: .rect(cornerRadius: 18, style: .continuous))
    .accessibilityElement(children: .combine)
  }
}
