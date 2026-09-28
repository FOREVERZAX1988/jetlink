import Foundation

/// Everything an app's screens show, kept from the server's events as the
/// iPhone's PhoneServer and ModelStore keep it, behind a lock instead of the
/// main actor, and handed out whole as JSON. The Android app draws these
/// snapshots (JetlinkAndroid); `android_snapshot.json` in the tests' fixtures
/// is one, which the Android app's own tests parse.
public final class AppSnapshot: @unchecked Sendable {
  /// How long a finished download stays on its row, as on the other apps.
  public static let terminalDownloadLinger: TimeInterval = 3
  /// The window the headline numbers cover, as on the iPhone.
  public static let recentWindow: TimeInterval = 10

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

  public init() {}

  public func reset() {
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

  public func serverStarted() {
    condition.lock()
    running = true
    bump()
    condition.unlock()
  }

  public func serverStopped() {
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

  public func apply(_ event: ControlEvent) {
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
      // with the report as the other apps share it
      var event = controlJSON(value)
      if var object = event as? [String: Any], let text = value.report?.text {
        object["report_text"] = text
        event = object
      }
      benchmark = event
    case .shutdownRequest(let value):
      shutdownRequest = controlJSON(value)
      shutdownRequests += 1
    case .hello, .reply, .unknown:
      return
    }
    bump()
  }

  /// Waits until the state is newer than `after`, or `timeout` passes, and
  /// returns it whole. `port` is where the server listens; `recent` the
  /// frames of the last `recentWindow` seconds, asked only while a comma is
  /// connected.
  public func snapshot(after: Int, timeout: TimeInterval, port: Int? = nil, recent: () -> StatsEvent? = { nil }) -> [String: Any] {
    condition.lock()
    defer { condition.unlock() }
    let deadline = Date().addingTimeInterval(max(0, timeout))
    while version <= after {
      if !condition.wait(until: deadline) { break }
    }
    let now = Date()
    for (sha, at) in finished where now.timeIntervalSince(at) >= AppSnapshot.terminalDownloadLinger {
      downloads[sha] = nil
      finished[sha] = nil
    }
    let rows = ModelRowBuilder.build(catalog: catalog, inventory: inventory, downloads: downloads, engine: engine, link: link)
    let recentStats = link.state == .connected ? recent() : nil
    return [
      "version": version,
      "running": running,
      "port": port ?? NSNull(),
      "server": controlJSON(server),
      "link": controlJSON(link),
      // what the connected comma's link is carried over, as the apps name it
      "medium": link.connectedMedium.map { ["name": $0.rawValue, "title": $0.title, "slow": $0.isSlow] as [String: Any] } ?? NSNull(),
      "engine": controlJSON(engine),
      "recent": controlJSON(recentStats),
      "history": history.map { ["at": $0.at.timeIntervalSince1970, "stats": controlJSON($0.stats)] as [String: Any] },
      "models": rows.map(AppSnapshot.row),
      "catalog": catalog.map { catalog -> Any in
        [
          "fetched_at": catalog.fetchedAt ?? NSNull(),
          "error": catalog.error ?? NSNull(),
          "default_ref": catalog.defaultRef,
          "count": catalog.models.count,
        ] as [String: Any]
      } ?? NSNull(),
      "disk": inventory.map { controlJSON($0.disk) } ?? NSNull(),
      "loaded": inventory?.loaded ?? NSNull(),
      "imports": imports.map { controlJSON($0) },
      "benchmark": benchmark,
      "shutdown_request": shutdownRequest,
      "shutdown_requests": shutdownRequests,
    ]
  }

  /// A Models row as the app draws it.
  public static func row(_ row: ModelRow) -> [String: Any] {
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
      "display_name": row.displayName,
      "ref": row.ref ?? NSNull(),
      "sha256": row.sha256 ?? NSNull(),
      "bytes": row.bytes ?? NSNull(),
      "build_time": row.buildTime ?? NSNull(),
      "status": status,
      "prepared_for": row.preparedFor.map { controlJSON($0) },
      "is_loaded": row.isLoaded,
      "is_default": row.isDefault,
      "is_requested_by_comma": row.isRequestedByComma,
      "is_local": row.isLocal,
      "is_orphan": row.isOrphan,
      "can_use": canUse,
    ]
  }
}

/// An Encodable event as the control protocol writes it: a snake_case
/// object, or null.
public func controlJSON<T: Encodable>(_ value: T?) -> Any {
  guard let value else { return NSNull() }
  let encoder = JSONEncoder()
  encoder.keyEncodingStrategy = .convertToSnakeCase
  guard let data = try? encoder.encode(value), let object = try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed]) else {
    return NSNull()
  }
  return object
}
