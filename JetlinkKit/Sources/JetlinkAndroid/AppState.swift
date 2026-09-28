import Foundation
import JetlinkKit
import JetlinkServer

/// Everything the Android app's screens show, kept from the server's events
/// as the iPhone's PhoneServer and ModelStore keep it, behind a lock instead
/// of the main actor. The app asks for a snapshot and draws it.
final class AppState: @unchecked Sendable {
  /// How long a finished download stays on its row, as on the other apps.
  static let terminalDownloadLinger: TimeInterval = 3
  /// The window the headline numbers cover, as on the iPhone.
  static let recentWindow: TimeInterval = 10

  private let condition = NSCondition()
  private var version = 1
  private var running = false
  private var server: ServerEvent?
  private var link: LinkEvent = .waiting
  private var engine: EngineEvent = .none
  private var catalog: CatalogEvent?
  private var inventory: InventoryEvent?
  private var downloads: [String: DownloadEvent] = [:]
  private var finished: [String: Date] = [:]
  private var imports: [ImportEvent] = []
  private var history: [StatsSample] = []
  private var benchmark: Any = NSNull()
  private var shutdownRequest: Any = NSNull()
  private var shutdownRequests = 0

  func reset() {
    condition.lock()
    server = nil
    link = .waiting
    engine = .none
    downloads = [:]
    finished = [:]
    imports = []
    history = []
    benchmark = NSNull()
    bump()
    condition.unlock()
  }

  func serverStarted() {
    condition.lock()
    running = true
    bump()
    condition.unlock()
  }

  func serverStopped() {
    condition.lock()
    running = false
    link = .waiting
    history = []
    bump()
    condition.unlock()
  }

  /// Under the lock.
  private func bump() {
    version += 1
    condition.broadcast()
  }

  func apply(_ event: ControlEvent) {
    condition.lock()
    defer { condition.unlock() }
    switch event {
    case .server(let value):
      server = value
    case .link(let value):
      link = value
      if value.state != .connected { history = [] }
    case .engine(let value):
      engine = value
    case .stats(let value):
      history = StatsSample.appending(value, to: history)
    case .inventory(let value):
      inventory = value
    case .catalog(let value):
      catalog = value
    case .download(let value):
      downloads[value.sha256] = value
      if ["done", "failed", "cancelled"].contains(value.state) {
        finished[value.sha256] = Date()
      } else {
        finished[value.sha256] = nil
      }
    case .importEvent(let value):
      if let index = imports.firstIndex(where: { $0.path == value.path }) {
        imports[index] = value
      } else {
        imports.append(value)
      }
    case .benchmark(let value):
      benchmark = object(value)
    case .shutdownRequest(let value):
      shutdownRequest = object(value)
      shutdownRequests += 1
    case .hello, .reply, .unknown:
      return
    }
    bump()
  }

  /// Waits until the state is newer than `after`, or `timeout` passes, and
  /// returns it whole.
  func snapshot(after: Int, timeout: TimeInterval, server embedded: EmbeddedServer?) -> [String: Any] {
    condition.lock()
    defer { condition.unlock() }
    let deadline = Date().addingTimeInterval(max(0, timeout))
    while version <= after {
      if !condition.wait(until: deadline) { break }
    }
    let now = Date()
    for (sha, at) in finished where now.timeIntervalSince(at) >= AppState.terminalDownloadLinger {
      downloads[sha] = nil
      finished[sha] = nil
    }
    let rows = ModelRowBuilder.build(catalog: catalog, inventory: inventory, downloads: downloads, engine: engine, link: link)
    let recent = link.state == .connected ? embedded?.server.recentStats(window: AppState.recentWindow) : nil
    return [
      "version": version,
      "running": running,
      "port": embedded?.server.port.map { Int($0) } ?? NSNull(),
      "server": object(server),
      "link": object(link),
      "engine": object(engine),
      "recent": object(recent),
      "history": history.map { ["at": $0.at.timeIntervalSince1970, "stats": object($0.stats)] as [String: Any] },
      "models": rows.map(AppState.row),
      "catalog": catalog.map { catalog -> Any in
        [
          "fetched_at": catalog.fetchedAt ?? NSNull(),
          "error": catalog.error ?? NSNull(),
          "default_ref": catalog.defaultRef,
          "count": catalog.models.count,
        ] as [String: Any]
      } ?? NSNull(),
      "disk": inventory.map { object($0.disk) } ?? NSNull(),
      "loaded": inventory?.loaded ?? NSNull(),
      "imports": imports.map { object($0) },
      "benchmark": benchmark,
      "shutdown_request": shutdownRequest,
      "shutdown_requests": shutdownRequests,
    ]
  }

  /// A Models row as the app draws it.
  static func row(_ row: ModelRow) -> [String: Any] {
    var status: [String: Any]
    switch row.status {
    case .unresolved: status = ["kind": "unresolved"]
    case .notDownloaded: status = ["kind": "not_downloaded"]
    case .downloading(let frac, let rate): status = ["kind": "downloading", "frac": frac, "rate_bps": rate]
    case .downloaded: status = ["kind": "downloaded"]
    case .preparing(let stage, let frac, let msg): status = ["kind": "preparing", "stage": stage, "frac": frac, "msg": msg]
    case .prepared: status = ["kind": "prepared"]
    case .loaded: status = ["kind": "loaded"]
    case .failed(let detail): status = ["kind": "failed", "detail": detail]
    }
    // The other apps' ModelStore.canUse: what "Use" may be pressed on.
    let canUse: Bool
    switch row.status {
    case .notDownloaded, .downloaded, .prepared, .failed: canUse = row.sha256 != nil || row.ref != nil
    case .unresolved, .downloading, .preparing, .loaded: canUse = false
    }
    return [
      "id": row.id,
      "name": row.name,
      "ref": row.ref ?? NSNull(),
      "sha256": row.sha256 ?? NSNull(),
      "bytes": row.bytes ?? NSNull(),
      "build_time": row.buildTime ?? NSNull(),
      "status": status,
      "prepared_for": row.preparedFor.map { object($0) },
      "is_loaded": row.isLoaded,
      "is_default": row.isDefault,
      "is_requested_by_comma": row.isRequestedByComma,
      "is_local": row.isLocal,
      "is_orphan": row.isOrphan,
      "can_use": canUse,
    ]
  }
}
