import Foundation
import JetlinkKit
import JetlinkUI
import SwiftUI

/// Everything the dashboard draws, as plain values, so the view is a function
/// of it and previews need no server.
struct DashboardState: Equatable {
  var runState: ServerRunState = .serving
  var link: LinkEvent = .waiting
  var engine: EngineEvent = .none
  /// The engine's model by name, when the catalog knows it.
  var modelName: String?
  /// The last ten seconds of frames: steadier than one second for a number
  /// read at a glance, and still quick to show a change.
  var recent: StatsEvent?
  /// One summary a second, for the trend.
  var history: [StatsSample] = []
  /// What the comma's JetlinkEndpoint should say to reach this phone.
  var endpoint: String?
  var health = DeviceHealth()
  /// The model a comma nobody changed asks for, for the empty state.
  var defaultModel: ModelRow?
  var hasPreparedModel = false
  /// Where the model runs, in words: "Neural Engine and GPU".
  var computeSummary = "Neural Engine and GPU"
  /// iOS refuses Jetlink the local network, so the comma cannot connect.
  var localNetworkDenied = false

  var isServingFrames: Bool {
    link.state == .connected && engine.state == .ready && recent != nil
  }

  /// What the big card in the middle shows.
  enum Hero: Equatable {
    /// The comma is connected to a loaded model; nil until its first frames land.
    case budget(StatsEvent?)
    case progress(EngineEvent)
    case waiting
    case noModel
    case failed(String)
  }

  var hero: Hero {
    if case .failed(let reason) = runState {
      return .failed(reason)
    }
    switch engine.state {
    case .building, .loading:
      return .progress(engine)
    case .failed:
      return .failed(engine.detail.isEmpty ? "The model could not be prepared." : engine.detail)
    case .ready:
      return link.state == .connected ? .budget(recent) : .waiting
    case .none:
      return hasPreparedModel ? .waiting : .noModel
    }
  }

  /// The one line at the top: is the comma getting the big model?
  var headline: (title: String, detail: String, tone: StatusBadge.Tone) {
    let model = modelName ?? "the model"
    switch runState {
    case .failed:
      return ("Server stopped", "The comma drives on its small model until Jetlink starts again.", .bad)
    case .starting:
      return ("Starting", "", .info)
    case .stopping, .stopped:
      return ("Stopped", "The comma drives on its small model.", .neutral)
    case .serving:
      break
    }
    switch engine.state {
    case .building:
      return ("Preparing \(model)", "The comma drives on its small model until this finishes.", .info)
    case .loading:
      return ("Loading \(model)", "The comma drives on its small model until this finishes.", .info)
    case .failed:
      return ("Model failed", "The comma drives on its small model.", .bad)
    case .ready, .none:
      break
    }
    if localNetworkDenied && link.state != .connected {
      return ("Local Network is off", "The comma cannot connect until Jetlink is allowed on the local network.", .bad)
    }
    switch link.state {
    case .connected:
      if engine.state == .ready {
        return ("Serving \(model)", link.peer.map { "To the comma at \(Self.host($0))" } ?? "", .good)
      }
      return ("Comma connected", "No model is loaded, so the comma drives on its small model.", .warning)
    case .waiting:
      return ("Waiting for the comma", engine.state == .ready ? "\(model) is ready." : "", .neutral)
    case .disconnected:
      return ("Comma disconnected", link.detail.isEmpty ? "" : Self.sentence(link.detail), .warning)
    }
  }

  /// "10.0.0.1" from "10.0.0.1:53412": the comma's port is noise here.
  static func host(_ peer: String) -> String {
    guard let colon = peer.lastIndex(of: ":") else { return peer }
    return String(peer[..<colon])
  }

  static func sentence(_ text: String) -> String {
    guard let first = text.first else { return text }
    let capitalized = first.uppercased() + text.dropFirst()
    return capitalized.hasSuffix(".") ? capitalized : capitalized + "."
  }
}
