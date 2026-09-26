import JetlinkKit
import SwiftUI

/// The frame budget as a gauge: a 270° track as long as the 50 ms the comma
/// gives each frame, filled to the p99 frame time, with the room left as the
/// one big number inside. Made to be read at a glance from a phone in a car
/// mount, so it carries one number, one state and nothing else.
///
/// The fill carries the state (the same three the frame budget view uses);
/// the track is a lighter step of the same colour, so the state reads across
/// the whole ring, and the state is also spelled out underneath with its icon.
public struct HeadroomRing: View {
  let p99: Double?
  let lineWidth: CGFloat

  /// `p99` nil draws an empty track: no frames to measure yet.
  public init(p99: Double?, lineWidth: CGFloat = 18) {
    self.p99 = p99
    self.lineWidth = lineWidth
  }

  /// The ring's share of a circle; the gap sits at the bottom.
  static let sweep = 0.75

  public var body: some View {
    let room = p99.map { FrameBudgetView.Room(headroomMs: FrameBudgetView.budgetMs - $0) }
    let tint = room?.tone.color ?? .secondary
    let fraction = p99.map { min(max($0 / FrameBudgetView.budgetMs, 0), 1) } ?? 0
    ZStack {
      arc(to: HeadroomRing.sweep)
        .stroke(tint.opacity(p99 == nil ? 0.15 : 0.2), style: StrokeStyle(lineWidth: lineWidth, lineCap: .round))
      if p99 != nil {
        arc(to: HeadroomRing.sweep * fraction)
          .stroke(tint, style: StrokeStyle(lineWidth: lineWidth, lineCap: .round))
          .animation(.smooth(duration: 0.6), value: fraction)
      }
      ForEach(Array(stride(from: 10.0, to: FrameBudgetView.budgetMs, by: 10)), id: \.self) { ms in
        tick(at: ms / FrameBudgetView.budgetMs)
      }
      center(room: room)
      Text("\(Int(FrameBudgetView.budgetMs)) ms budget")
        .font(.caption.weight(.medium))
        .foregroundStyle(.secondary)
        .frame(maxHeight: .infinity, alignment: .bottom)
    }
    .padding(lineWidth / 2)
    .aspectRatio(1, contentMode: .fit)
    .accessibilityElement(children: .ignore)
    .accessibilityLabel(accessibilityText(room))
  }

  private func arc(to fraction: Double) -> some Shape {
    Circle()
      .trim(from: 0, to: fraction)
      .rotation(.degrees(135))
  }

  /// A hairline across the track every 10 ms, so the fill reads against a scale.
  private func tick(at fraction: Double) -> some View {
    GeometryReader { proxy in
      let radius = min(proxy.size.width, proxy.size.height) / 2
      let angle = Angle.degrees(135 + 360 * HeadroomRing.sweep * fraction)
      Capsule()
        .fill(.background.opacity(0.9))
        .frame(width: 2, height: lineWidth)
        .rotationEffect(angle + .degrees(90))
        .position(
          x: proxy.size.width / 2 + radius * cos(angle.radians),
          y: proxy.size.height / 2 + radius * sin(angle.radians))
    }
  }

  @ViewBuilder
  private func center(room: FrameBudgetView.Room?) -> some View {
    VStack(spacing: 2) {
      if let p99, let room {
        let headroom = FrameBudgetView.budgetMs - p99
        Text(abs(headroom).formatted(.number.precision(.fractionLength(1))))
          .font(.system(size: 64, weight: .semibold))
          .contentTransition(.numericText(value: headroom))
          .animation(.smooth, value: headroom)
          .minimumScaleFactor(0.5)
          .lineLimit(1)
        Text(headroom >= 0 ? "ms to spare" : "ms over")
          .font(.headline)
          .foregroundStyle(.secondary)
        Label(room.title, systemImage: room.symbol)
          .font(.subheadline.weight(.semibold))
          .foregroundStyle(room.tone.color)
          .padding(.top, 6)
      } else {
        Text("—")
          .font(.system(size: 64, weight: .semibold))
          .foregroundStyle(.tertiary)
        Text("no frames yet")
          .font(.headline)
          .foregroundStyle(.secondary)
      }
    }
    .padding(.horizontal, lineWidth * 2)
  }

  private func accessibilityText(_ room: FrameBudgetView.Room?) -> String {
    guard let p99, let room else { return "No frames measured yet" }
    return "\(FrameBudgetView.headroomText(p99: p99)) at p99 of a \(Int(FrameBudgetView.budgetMs)) millisecond budget. \(room.title)."
  }
}

#Preview("Room to spare") {
  HeadroomRing(p99: 31.6)
    .frame(width: 280)
    .padding()
}

#Preview("Tight and over") {
  HStack {
    HeadroomRing(p99: 44.2)
    HeadroomRing(p99: 58.3)
    HeadroomRing(p99: nil)
  }
  .padding()
}
