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
  let localNetwork: LocalNetworkAccess
  @ObservationIgnored private var launched = false

  init() {
    let settings = PhoneSettings()
    let server = PhoneServer(settings: settings)
    self.settings = settings
    self.server = server
    self.models = ModelStore(server: server)
    self.device = DeviceMonitor()
    self.network = NetworkInterfaces()
    self.localNetwork = LocalNetworkAccess()
  }

  /// The server starts with the app: an iPhone app has nothing else to do,
  /// and a comma may already be asking.
  func launch() {
    guard !launched else { return }
    launched = true
    server.start()
    localNetwork.check()
    models.refreshCatalog()
  }

  /// Everything the Status tab draws, from the stores.
  var status: StatusState {
    var state = StatusState()
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
    state.catalogUnavailable = models.catalog?.error != nil && (models.catalog?.models.isEmpty ?? true)
    state.localNetworkDenied = localNetwork.state == .denied
    return state
  }

  /// The list failed to load and nothing is cached: try again when the
  /// network changes or the app comes back, rather than leave it empty.
  func retryCatalogIfEmpty() {
    guard let catalog = models.catalog, catalog.error != nil, catalog.models.isEmpty else { return }
    models.refreshCatalog()
  }

  func modelName(_ sha256: String?) -> String? {
    guard let sha256 else { return nil }
    guard let row = models.rows.first(where: { $0.sha256 == sha256 }) else { return nil }
    // A model the comma sent that no catalog names has only its hash.
    return row.isOrphan ? "Uploaded Model" : row.displayName
  }
}
