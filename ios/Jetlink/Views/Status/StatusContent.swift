import JetlinkKit
import JetlinkUI
import SwiftUI

/// The Status tab's cards, most important first: the headroom, where the time
/// goes, the last two minutes, then the link and the phone. On its side the
/// phone shows the ring on the left and the rest beside it. No UIKit here, so
/// it renders anywhere SwiftUI does.
struct StatusContent: View {
  let state: StatusState
  var landscape = false
  var actions = StatusActions()
  /// Off only for snapshots: ImageRenderer draws nothing inside a ScrollView.
  var scrolls = true

  static let margin: CGFloat = 16
  static let spacing: CGFloat = 16

  var body: some View {
    if scrolls {
      ScrollView {
        layout
          .padding(.bottom, 24)
      }
      .contentMargins(.horizontal, StatusContent.margin, for: .scrollContent)
      .background(Color.groupedBackground)
    } else {
      layout
        .padding(.horizontal, StatusContent.margin)
        .background(Color.groupedBackground)
    }
  }

  @ViewBuilder
  private var layout: some View {
    if landscape {
      HStack(alignment: .top, spacing: StatusContent.spacing) {
        HeroCard(state: state, compact: true, actions: actions)
          .frame(width: 330)
        VStack(spacing: StatusContent.spacing) {
          if let recent = state.recent, state.isServingFrames {
            latency(recent, compact: true)
            link(recent)
          } else {
            phone
          }
        }
      }
    } else {
      VStack(spacing: StatusContent.spacing) {
        HeroCard(state: state, actions: actions)
        if let recent = state.recent, state.isServingFrames {
          latency(recent)
          if state.history.count > 1 {
            history
          }
          SectionHeader("Link")
          link(recent)
        }
        SectionHeader("iPhone")
        phone
      }
    }
  }

  private func latency(_ recent: StatsEvent, compact: Bool = false) -> some View {
    SummaryCard(title: "Latency", systemImage: "timer", tint: .blue, trailing: "Last 10 s") {
      LatencyBreakdown(stats: recent, compact: compact)
    }
  }

  private var history: some View {
    SummaryCard(title: "History", systemImage: "chart.xyaxis.line", tint: .purple, trailing: "2 min") {
      LatencyHistoryChart(history: state.history)
    }
  }

  private func link(_ recent: StatsEvent) -> some View {
    MetricGrid {
      GridRow {
        MetricTile(
          title: "Frame Rate", systemImage: "speedometer", tint: .teal,
          value: recent.fps.formatted(.number.precision(.fractionLength(1))), unit: "fps",
          note: recent.fps < 18 ? "Below 20" : nil, noteTone: .orange)
        MetricTile(
          title: "Slow Frames", systemImage: "tortoise.fill", tint: .pink,
          value: recent.slow.formatted(), note: recent.slow > 0 ? "Over 60 ms" : nil, noteTone: .red)
      }
    }
  }

  private var phone: some View {
    let health = state.health
    return MetricGrid {
      GridRow {
        MetricTile(
          title: "Temperature", systemImage: health.thermal.symbol, tint: .orange,
          value: health.thermal.title, note: health.thermal.note, noteTone: health.thermal.tone)
        MetricTile(
          title: "Battery", systemImage: health.batterySymbol, tint: .green,
          value: health.batteryValue, unit: health.batteryLevel == nil ? nil : "%",
          note: health.powerText, noteTone: health.batteryTone)
      }
    }
  }
}

/// A section title over a group of cards, as Health and Fitness set them.
struct SectionHeader: View {
  let title: String

  init(_ title: String) {
    self.title = title
  }

  var body: some View {
    Text(title)
      .font(.title3.weight(.bold))
      .frame(maxWidth: .infinity, alignment: .leading)
      .padding(.top, 8)
      .padding(.bottom, -4)
      .accessibilityAddTraits(.isHeader)
  }
}
