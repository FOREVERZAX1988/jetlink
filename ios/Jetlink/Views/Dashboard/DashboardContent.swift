import JetlinkKit
import JetlinkUI
import SwiftUI

/// The dashboard's cards, laid out for a phone in a car mount: the state and
/// the budget first, where a glance lands, then where the time goes, then the
/// last two minutes, then the phone itself. Two columns when the phone lies on
/// its side. No UIKit here, so it renders anywhere SwiftUI does.
struct DashboardContent: View {
  let state: DashboardState
  var landscape = false
  var onUseDefault: () -> Void = {}
  var onOpenModels: () -> Void = {}
  var onRetry: () -> Void = {}
  var onOpenSettings: () -> Void = {}

  /// Off only for snapshots: ImageRenderer draws nothing inside a ScrollView.
  var scrolls = true

  var body: some View {
    if scrolls {
      ScrollView { cards }
        .background(Color.groupedBackground)
    } else {
      cards
        .background(Color.groupedBackground)
    }
  }

  private var cards: some View {
    Group {
      if landscape {
        // A phone on its side is about 350 points tall under the bar: the
        // ring takes the left and fits the height, the rest scrolls beside it.
        HStack(alignment: .top, spacing: 16) {
          HeroCard(state: state, compact: true, onUseDefault: onUseDefault, onOpenModels: onOpenModels, onRetry: onRetry, onOpenSettings: onOpenSettings)
            .frame(width: 330)
          VStack(spacing: 12) {
            headline
            details
          }
          .frame(maxWidth: .infinity)
        }
      } else {
        VStack(spacing: 16) {
          headline
          hero
          details
        }
      }
    }
    .padding(.horizontal, 16)
    .padding(.bottom, 24)
  }

  private var hero: some View {
    HeroCard(state: state, onUseDefault: onUseDefault, onOpenModels: onOpenModels, onRetry: onRetry, onOpenSettings: onOpenSettings)
  }

  @ViewBuilder
  private var details: some View {
    if let recent = state.recent, state.isServingFrames {
      Card(title: "Where the Time Goes", systemImage: "timer") {
        FrameStageBar(stats: recent)
        FrameStageTable(stages: recent.stages)
      }
      if state.history.count > 1 {
        Card(title: "Last Two Minutes", systemImage: "chart.xyaxis.line") {
          FrameTimeChart(history: state.history, showsTitle: false)
        }
      }
      kpis(recent)
    }
    phone
  }

  private var headline: some View {
    let (title, detail, tone) = state.headline
    return HStack(alignment: .firstTextBaseline, spacing: 10) {
      Circle()
        .fill(tone.color)
        .frame(width: 12, height: 12)
        .alignmentGuide(.firstTextBaseline) { $0[.bottom] - 2 }
      VStack(alignment: .leading, spacing: 2) {
        Text(title)
          .font(.title2.weight(.bold))
          .lineLimit(2)
          .minimumScaleFactor(0.8)
        if !detail.isEmpty {
          Text(detail)
            .font(.subheadline)
            .foregroundStyle(.secondary)
        }
      }
      Spacer(minLength: 0)
    }
    .padding(.horizontal, 4)
    .padding(.top, 4)
    .accessibilityElement(children: .combine)
  }

  private func kpis(_ recent: StatsEvent) -> some View {
    Grid(horizontalSpacing: 12, verticalSpacing: 12) {
      GridRow {
        StatTile(
          label: "Rate", value: "\(recent.fps.formatted(.number.precision(.fractionLength(1)))) fps",
          detail: recent.fps < 18 ? "The comma sends 20" : "Frames a second", systemImage: "speedometer",
          tone: recent.fps < 18 ? .orange : nil)
        StatTile(
          label: "Slow frames", value: recent.slow.formatted(),
          detail: "Over 60 ms, last 10 s", systemImage: "tortoise",
          tone: recent.slow > 0 ? .red : nil)
      }
      GridRow {
        StatTile(label: "Model", value: FrameBudgetView.ms(recent.gpuMs.mean), detail: state.computeSummary, systemImage: "cpu")
        StatTile(label: "Frames", value: recent.frames.formatted(), detail: "This connection", systemImage: "film.stack")
      }
    }
    // rows as tall as their tallest tile, and no taller
    .fixedSize(horizontal: false, vertical: true)
  }

  private var phone: some View {
    let health = state.health
    return Grid(horizontalSpacing: 12, verticalSpacing: 12) {
      GridRow {
        StatTile(
          label: "Temperature", value: health.thermal.title, detail: health.thermal.detail, systemImage: health.thermal.symbol, tone: health.thermal.tone)
        StatTile(label: "Battery", value: health.batteryText, detail: health.powerText, systemImage: health.batterySymbol, tone: health.batteryTone)
      }
    }
    .fixedSize(horizontal: false, vertical: true)
  }
}
