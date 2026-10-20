#if canImport(SwiftUI)
  import JetlinkKit
  import SwiftUI

  /// A frame's stages as a small table, two to a row: the key to the stacked
  /// bar, carrying every value so the bar never has to be read on its own. For
  /// a phone, where `FrameStageLegend`'s single row does not fit.
  ///
  /// Both halves of the budget can be here at once: the host's four stages,
  /// which begin when the frame arrives, and the comma's six, which end when it
  /// is sent. They are two views of one 50 ms rather than two totals, and a slow
  /// frame only says which end of the cable it was with both.
  public struct FrameStageTable: View {
    let sections: [Section]

    /// The host's half: staging the inputs, the model, the rest of the run,
    /// and sending the reply.
    public init(stages: StatsEvent.Stages) {
      sections = [FrameStageTable.host(stages)]
    }

    /// The comma's half, when a big model is driving there.
    public init(comma: CommaStages) {
      sections = [FrameStageTable.comma(comma)]
    }

    /// Both, the comma's left out while it has said nothing recent.
    public init(stages: StatsEvent.Stages, comma: CommaStages?) {
      var built = [FrameStageTable.host(stages)]
      if let comma {
        built.append(FrameStageTable.comma(comma))
      }
      sections = built
    }

    private static func host(_ stages: StatsEvent.Stages) -> Section {
      Section(name: "Host", lines: FrameStage.allCases.map { stage in
        Line(title: stage.title, color: stage.color, ms: stage.value(stages), detail: stage.detail)
      })
    }

    private static func comma(_ stages: CommaStages) -> Section {
      Section(name: "Comma", lines: CommaStage.allCases.map { stage in
        Line(title: stage.title, color: stage.color, ms: stage.value(stages), detail: stage.detail)
      })
    }

    public var body: some View {
      VStack(alignment: .leading, spacing: 10) {
        ForEach(sections) { section in
          VStack(alignment: .leading, spacing: 6) {
            // One section needs no heading: the view around it already says
            // what it is about.
            if sections.count > 1 {
              Text(section.name)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            }
            Grid(alignment: .leading, horizontalSpacing: 10, verticalSpacing: 8) {
              ForEach(Array(stride(from: 0, to: section.lines.count, by: 2)), id: \.self) { start in
                GridRow {
                  ForEach(section.lines[start..<min(start + 2, section.lines.count)]) { line in
                    RoundedRectangle(cornerRadius: 2)
                      .fill(line.color)
                      .frame(width: 10, height: 10)
                    Text(line.title)
                      .foregroundStyle(.secondary)
                    Text(FrameBudgetView.ms(line.ms))
                      .monospacedDigit()
                      .gridColumnAlignment(.trailing)
                      .padding(.trailing, 12)
                  }
                }
              }
            }
            .font(.subheadline)
            .accessibilityElement(children: .ignore)
            .accessibilityLabel(section.accessibilityText)
          }
        }
      }
    }

    /// One stage's row.
    struct Line: Identifiable {
      let title: String
      let color: Color
      let ms: Double
      let detail: String
      var id: String { title }
    }

    struct Section: Identifiable {
      let name: String
      let lines: [Line]
      var id: String { name }

      /// Read out whole, since the cells are all it says.
      var accessibilityText: String {
        let parts = lines.map { "\($0.title) \(FrameBudgetView.ms($0.ms))" }
        return "\(name): " + parts.joined(separator: ", ")
      }
    }
  }

  /// The comma's six stages, in the order a frame goes through them
  /// (openpilot/stages.py). Each begins where the last ended, and the six sum to
  /// the frame's own period.
  public enum CommaStage: CaseIterable, Identifiable, Sendable {
    case capture, warp, send, replyWait, publish, overhead

    public var id: Self { self }

    public var title: String {
      switch self {
      case .capture: "Read"
      case .warp: "Warp"
      case .send: "Send"
      case .replyWait: "Wait"
      case .publish: "Publish"
      case .overhead: "Other"
      }
    }

    public var detail: String {
      switch self {
      case .capture: "From the sensor's read-out to this frame's turn"
      case .warp: "openpilot's warp, on the comma's GPU"
      case .send: "Writing the request to the link"
      case .replyWait: "Waiting on the host's answer"
      case .publish: "Parsing the outputs and handing them to modeld"
      case .overhead: "The rest of the frame, to the next one's start"
      }
    }

    public func value(_ stages: CommaStages) -> Double {
      switch self {
      case .capture: stages.capture
      case .warp: stages.warp
      case .send: stages.send
      case .replyWait: stages.replyWait
      case .publish: stages.publish
      case .overhead: stages.overhead
      }
    }

    /// Six steps that stay apart under colour blindness, with the warp and the
    /// wait farthest apart, since those are the two the eye is deciding
    /// between: the comma's GPU or the host. The remainder is grey, because it
    /// is not work anyone chose to do.
    public var color: Color {
      switch self {
      case .capture: Color(light: 0x7E57C2, dark: 0x9575CD)
      case .warp: Color(light: 0xEB6834, dark: 0xD95926)
      case .send: Color(light: 0xEDA100, dark: 0xC98500)
      case .replyWait: Color(light: 0x2A78D6, dark: 0x3987E5)
      case .publish: Color(light: 0x1BAF7A, dark: 0x199E70)
      case .overhead: Color(light: 0x8A8F98, dark: 0x9AA0A9)
      }
    }
  }
#endif
