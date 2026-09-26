import Foundation
import JetlinkKit
import Observation

/// The composition root. One of these exists for the life of the app.
@MainActor
@Observable
final class AppModel {
  let settings: PhoneSettings
  let server: PhoneServer
  let models: ModelStore
  let device: DeviceMonitor
  let network: NetworkInterfaces
  @ObservationIgnored private var launched = false

  init() {
    let settings = PhoneSettings()
    let server = PhoneServer(settings: settings)
    self.settings = settings
    self.server = server
    self.models = ModelStore(server: server)
    self.device = DeviceMonitor()
    self.network = NetworkInterfaces()
  }

  /// The server starts with the app: an iPhone app has nothing else to do,
  /// and a comma may already be asking.
  func launch() {
    guard !launched else { return }
    launched = true
    server.start()
    models.refreshCatalog()
  }

  /// Everything the dashboard draws, from the stores.
  var dashboard: DashboardState {
    var state = DashboardState()
    state.runState = server.runState
    state.link = server.link
    state.engine = server.engine
    state.modelName = modelName(server.engine.sha256)
    state.recent = server.recent
    state.history = server.statsHistory
    if let address = network.preferred, let port = server.port {
      state.endpoint = "\(address.address):\(port)"
    }
    state.health = device.health
    state.defaultModel = models.rows.first { $0.isDefault }
    state.hasPreparedModel = !(models.inventory?.artifacts.filter(\.current).isEmpty ?? true)
    state.computeSummary = settings.device.title
    return state
  }

  func modelName(_ sha256: String?) -> String? {
    guard let sha256 else { return nil }
    return models.rows.first { $0.sha256 == sha256 }?.displayName
  }
}
