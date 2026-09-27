import SwiftUI

#if canImport(UIKit)
  import UIKit
#else
  import AppKit
#endif

extension Color {
  /// The screen behind the cards: grouped, as Health and Settings draw it.
  static var groupedBackground: Color {
    #if canImport(UIKit)
      Color(uiColor: .systemGroupedBackground)
    #else
      Color(
        nsColor: NSColor(name: nil) {
          $0.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua ? .black : NSColor(srgbRed: 0.949, green: 0.949, blue: 0.969, alpha: 1)
        })
    #endif
  }

  /// A card on the grouped background.
  static var cardBackground: Color {
    #if canImport(UIKit)
      Color(uiColor: .secondarySystemGroupedBackground)
    #else
      Color(
        nsColor: NSColor(name: nil) {
          $0.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua ? NSColor(srgbRed: 0.11, green: 0.11, blue: 0.118, alpha: 1) : .white
        })
    #endif
  }
}

/// The corner every card shares, so they read as one family.
let cardCornerRadius: CGFloat = 26

/// A card in the Health style: a tinted symbol and title, what it covers on
/// the right, then the content. Content, not a control, so no glass.
struct SummaryCard<Content: View>: View {
  let title: String
  let systemImage: String
  var tint: Color = .accentColor
  var trailing: String?
  @ViewBuilder var content: Content

  var body: some View {
    VStack(alignment: .leading, spacing: 14) {
      HStack(alignment: .firstTextBaseline) {
        Label(title, systemImage: systemImage)
          .font(.subheadline.weight(.semibold))
          .foregroundStyle(tint)
        Spacer()
        if let trailing {
          Text(trailing)
            .font(.subheadline)
            .foregroundStyle(.secondary)
        }
      }
      content
    }
    .padding(16)
    .frame(maxWidth: .infinity, alignment: .leading)
    .background(Color.cardBackground, in: .rect(cornerRadius: cardCornerRadius, style: .continuous))
  }
}

/// One number with its unit, under a tinted title: the Health app's metric tile.
struct MetricTile: View {
  let title: String
  let systemImage: String
  let tint: Color
  let value: String
  var unit: String?
  var note: String?
  var noteTone: Color?

  var body: some View {
    VStack(alignment: .leading, spacing: 6) {
      Label(title, systemImage: systemImage)
        .font(.subheadline.weight(.semibold))
        .foregroundStyle(tint)
        .lineLimit(1)
      HStack(alignment: .firstTextBaseline, spacing: 3) {
        Text(value)
          .font(.system(.title, design: .rounded, weight: .semibold))
          .contentTransition(.numericText())
        if let unit {
          Text(unit)
            .font(.system(.subheadline, design: .rounded, weight: .semibold))
            .foregroundStyle(.secondary)
        }
      }
      .lineLimit(1)
      .minimumScaleFactor(0.6)
      Text(note ?? " ")
        .font(.footnote)
        .foregroundStyle(noteTone ?? .secondary)
        .lineLimit(1)
    }
    .padding(14)
    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    .background(Color.cardBackground, in: .rect(cornerRadius: cardCornerRadius, style: .continuous))
    .accessibilityElement(children: .combine)
  }
}

/// Tiles two to a row, each row as tall as its tallest tile.
struct MetricGrid<Content: View>: View {
  @ViewBuilder var content: Content

  var body: some View {
    Grid(horizontalSpacing: 12, verticalSpacing: 12) {
      content
    }
    .fixedSize(horizontal: false, vertical: true)
  }
}

/// A label over a number in milliseconds, three or so to a row under a ring
/// or a verdict.
struct Figure: View {
  let label: String
  let ms: Double?
  var tone: Color = .primary

  init(_ label: String, ms: Double?, tone: Color = .primary) {
    self.label = label
    self.ms = ms
    self.tone = tone
  }

  var body: some View {
    VStack(spacing: 2) {
      Text(label)
        .font(.footnote.weight(.medium))
        .foregroundStyle(.secondary)
      HStack(alignment: .firstTextBaseline, spacing: 2) {
        Text(ms.map { $0.formatted(.number.precision(.fractionLength(1))) } ?? "--")
          .font(.system(.title3, design: .rounded, weight: .semibold))
          .foregroundStyle(tone)
          .contentTransition(.numericText(value: ms ?? 0))
        Text("ms")
          .font(.system(.footnote, design: .rounded, weight: .semibold))
          .foregroundStyle(.secondary)
      }
    }
    .frame(maxWidth: .infinity)
    .accessibilityElement(children: .combine)
  }
}
