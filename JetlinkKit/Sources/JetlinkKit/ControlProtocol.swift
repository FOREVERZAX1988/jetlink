import Foundation

// The control protocol, version 1. One JSON object per line, UTF-8, newline terminated.
// Client to server: {"id": <int>, "cmd": "<name>", ...arguments}
// Server to client: {"event": "<name>", "t": <float seconds>, ...}
//
// The Mac app hears these from the Python server over a socket. The iPhone
// app runs the server in process and hears the same values without the JSON.

public enum EngineState: String, Codable, Sendable {
  case none, building, loading, ready, failed
}

public enum LinkState: String, Codable, Sendable {
  case waiting, connected, disconnected
}

public struct HelloEvent: Codable, Sendable, Equatable {
  public let protocolVersion: Int
  public let pid: Int32
  public let version: String
  public let python: String
  public let platform: String
  public let cache: String
  public let transport: String
  public let port: Int?

  public init(protocolVersion: Int, pid: Int32, version: String, python: String, platform: String, cache: String, transport: String, port: Int?) {
    self.protocolVersion = protocolVersion
    self.pid = pid
    self.version = version
    self.python = python
    self.platform = platform
    self.cache = cache
    self.transport = transport
    self.port = port
  }

  // "protocol" carries no underscore, so the snake case strategy leaves it alone.
  enum CodingKeys: String, CodingKey {
    case protocolVersion = "protocol"
    case pid, version, python, platform, cache, transport, port
  }
}

public struct ServerEvent: Codable, Sendable, Equatable {
  public let state: String
  public let detail: String
  public let backend: String?
  public let runtimeVersion: String?
  public let device: String?

  public init(state: String, detail: String, backend: String?, runtimeVersion: String?, device: String?) {
    self.state = state
    self.detail = detail
    self.backend = backend
    self.runtimeVersion = runtimeVersion
    self.device = device
  }
}

public struct LinkEvent: Codable, Sendable, Equatable {
  public let state: LinkState
  public let detail: String
  public let peer: String?
  /// How a connected link is carried, as `LinkMedium` names it; nil from a
  /// server older than the field, or before it can tell.
  public let medium: String?

  public init(state: LinkState, detail: String, peer: String?, medium: String? = nil) {
    self.state = state
    self.detail = detail
    self.peer = peer
    self.medium = medium
  }

  public static let waiting = LinkEvent(state: .waiting, detail: "", peer: nil)

  public var linkMedium: LinkMedium? { medium.flatMap(LinkMedium.init(rawValue:)) }

  /// What a connected link is carried over, for display: the server's word,
  /// or for a server or comma older than the field, a guess from the peer.
  /// The Python server's USB host says "usb", and a peer on the comma's cable
  /// network is a phone's cable: USB of unknown speed. Anything else is TCP.
  public var connectedMedium: LinkMedium? {
    guard state == .connected else { return nil }
    if let linkMedium { return linkMedium }
    guard let peer, !peer.isEmpty, peer != "usb" else { return .usb }
    return LinkMedium(tcpPeer: peer)
  }
}

/// How the comma's link is carried: the USB generation its controller
/// negotiated, USB of unknown speed, or TCP. The comma's hello says which
/// (`Transport.link_info` in Python), since only its end always knows: a
/// phone's cable is TCP over USB. Names and mapping are `Pinned`.
public enum LinkMedium: String, Codable, Sendable, CaseIterable {
  case usb3, usb2, usb1, usb, tcp

  /// The comma's cable network, "192.168.60.", from its end's address.
  public static let cableNetwork = String(Pinned.cableAddress[...Pinned.cableAddress.lastIndex(of: ".")!])

  /// From a speed as Linux names it: super-speed, high-speed and so on.
  public init(usbSpeed: String?) {
    self = usbSpeed.flatMap { Pinned.usbSpeedMedia[$0] }.flatMap(LinkMedium.init(rawValue:)) ?? .usb
  }

  /// A TCP link, from its peer ("host:port") before a hello says more: the
  /// comma's cable address is a phone's USB cable, of unknown speed. The same
  /// rule as the Python transport's `on_the_cable`.
  public init(tcpPeer peer: String) {
    self = peer.hasPrefix(Pinned.cableAddress + ":") ? .usb : .tcp
  }

  /// From a hello's `client.link`, or nil when it names none.
  public init?(link: [String: Any]?) {
    switch link?["kind"] as? String {
    case "usb", "cable": self.init(usbSpeed: link?["usb_speed"] as? String)
    case "tcp": self = .tcp
    default: return nil
    }
  }

  public var title: String {
    switch self {
    case .usb3: "USB 3"
    case .usb2: "USB 2"
    case .usb1: "USB 1"
    case .usb: "USB"
    case .tcp: "TCP"
    }
  }

  /// A frame is about 460 KB: around 1 ms on USB 3, around 11 ms on USB 2,
  /// enough to cost frames and bring the comma near its soft disable.
  public var isSlow: Bool { self == .usb2 || self == .usb1 }

  /// What to do about a slow link, in a sentence; nil when it is fast enough.
  public var advice: String? {
    isSlow ? "\(title) costs about 10 ms a frame more than USB 3. Use a USB 3 cable and port." : nil
  }
}

public struct EngineEvent: Codable, Sendable, Equatable {
  public let state: EngineState
  public let sha256: String?
  public let detail: String
  public let stage: String?
  public let frac: Double
  public let msg: String
  public let loadOnly: Bool

  public init(state: EngineState, sha256: String?, detail: String, stage: String?, frac: Double, msg: String, loadOnly: Bool) {
    self.state = state
    self.sha256 = sha256
    self.detail = detail
    self.stage = stage
    self.frac = frac
    self.msg = msg
    self.loadOnly = loadOnly
  }

  public static let none = EngineEvent(state: .none, sha256: nil, detail: "", stage: nil, frac: 0, msg: "", loadOnly: false)
}

public struct StatsEvent: Codable, Sendable, Equatable {
  public struct Total: Codable, Sendable, Equatable {
    public let mean: Double
    public let p99: Double
    public let max: Double

    public init(mean: Double, p99: Double, max: Double) {
      self.mean = mean
      self.p99 = p99
      self.max = max
    }
  }

  /// Means that add up to `servedMs.mean`: staging the inputs, the model run,
  /// the rest of the run, and sending the reply.
  public struct Stages: Codable, Sendable, Equatable {
    public let queue: Double
    public let gpu: Double
    public let other: Double
    public let send: Double

    public init(queue: Double, gpu: Double, other: Double, send: Double) {
      self.queue = queue
      self.gpu = gpu
      self.other = other
      self.send = send
    }
  }

  public struct Mean: Codable, Sendable, Equatable {
    public let mean: Double

    public init(mean: Double) {
      self.mean = mean
    }
  }

  public let frames: Int
  public let fps: Double
  /// From a frame's arrival to its reply leaving.
  public let servedMs: Total
  public let stagesMs: Stages
  public let slow: Int
  public let windowS: Double
  /// From a frame's arrival to its reply being ready, without the send: what
  /// `slow` counts against. Nil from a server that does not send it.
  public let totalMs: Total?
  /// The model run, as the backend times it.
  public let gpuMs: Mean?

  public init(
    frames: Int, fps: Double, servedMs: Total, stagesMs: Stages, slow: Int, windowS: Double, totalMs: Total? = nil, gpuMs: Mean? = nil
  ) {
    self.frames = frames
    self.fps = fps
    self.servedMs = servedMs
    self.stagesMs = stagesMs
    self.slow = slow
    self.windowS = windowS
    self.totalMs = totalMs
    self.gpuMs = gpuMs
  }
}

public struct InventoryModel: Codable, Sendable, Equatable, Identifiable {
  public var id: String { sha256 }
  public let sha256: String
  public let bytes: Int64
  public let path: String
  public let name: String?
  public let ref: String?

  public init(sha256: String, bytes: Int64, path: String, name: String?, ref: String?) {
    self.sha256 = sha256
    self.bytes = bytes
    self.path = path
    self.name = name
    self.ref = ref
  }
}

public struct InventoryArtifact: Codable, Sendable, Equatable, Identifiable {
  public var id: String { key }
  public let sha256: String
  public let key: String
  public let path: String
  public let bytes: Int64
  public let backend: String
  public let runtimeVersion: String?
  public let device: String
  public let builtAt: String?
  public let buildSeconds: Double?
  public let checkpoint: String?
  public let current: Bool

  public init(
    sha256: String, key: String, path: String, bytes: Int64, backend: String, runtimeVersion: String?, device: String, builtAt: String?, buildSeconds: Double?,
    checkpoint: String?, current: Bool
  ) {
    self.sha256 = sha256
    self.key = key
    self.path = path
    self.bytes = bytes
    self.backend = backend
    self.runtimeVersion = runtimeVersion
    self.device = device
    self.builtAt = builtAt
    self.buildSeconds = buildSeconds
    self.checkpoint = checkpoint
    self.current = current
  }
}

public struct InventoryDisk: Codable, Sendable, Equatable {
  public let modelsBytes: Int64
  public let enginesBytes: Int64
  public let freeBytes: Int64

  public init(modelsBytes: Int64, enginesBytes: Int64, freeBytes: Int64) {
    self.modelsBytes = modelsBytes
    self.enginesBytes = enginesBytes
    self.freeBytes = freeBytes
  }
}

public struct InventoryEvent: Codable, Sendable, Equatable {
  public let loaded: String?
  public let lastLoaded: String?
  public let models: [InventoryModel]
  public let artifacts: [InventoryArtifact]
  public let disk: InventoryDisk

  public init(loaded: String?, lastLoaded: String?, models: [InventoryModel], artifacts: [InventoryArtifact], disk: InventoryDisk) {
    self.loaded = loaded
    self.lastLoaded = lastLoaded
    self.models = models
    self.artifacts = artifacts
    self.disk = disk
  }
}

public struct CatalogModel: Codable, Sendable, Equatable, Identifiable {
  public var id: String { ref }
  public let name: String
  public let shortName: String
  public let ref: String
  public let buildTime: String
  public let index: Int
  public let sha256: String?
  public let bytes: Int64?

  public init(name: String, shortName: String, ref: String, buildTime: String, index: Int, sha256: String?, bytes: Int64?) {
    self.name = name
    self.shortName = shortName
    self.ref = ref
    self.buildTime = buildTime
    self.index = index
    self.sha256 = sha256
    self.bytes = bytes
  }
}

public struct CatalogEvent: Codable, Sendable, Equatable {
  public let fetchedAt: Double?
  public let url: String
  public let defaultRef: String
  public let error: String?
  public let models: [CatalogModel]

  public init(fetchedAt: Double?, url: String, defaultRef: String, error: String?, models: [CatalogModel]) {
    self.fetchedAt = fetchedAt
    self.url = url
    self.defaultRef = defaultRef
    self.error = error
    self.models = models
  }
}

public struct DownloadEvent: Codable, Sendable, Equatable {
  public let sha256: String
  public let ref: String?
  public let state: String
  public let frac: Double
  public let bytes: Int64
  public let total: Int64
  public let rateBps: Double
  public let detail: String
  public let source: String?

  public init(sha256: String, ref: String?, state: String, frac: Double, bytes: Int64, total: Int64, rateBps: Double, detail: String, source: String?) {
    self.sha256 = sha256
    self.ref = ref
    self.state = state
    self.frac = frac
    self.bytes = bytes
    self.total = total
    self.rateBps = rateBps
    self.detail = detail
    self.source = source
  }
}

public struct ImportEvent: Codable, Sendable, Equatable {
  public let path: String
  public let state: String
  public let frac: Double
  public let sha256: String?
  public let detail: String

  public init(path: String, state: String, frac: Double, sha256: String?, detail: String) {
    self.path = path
    self.state = state
    self.frac = frac
    self.sha256 = sha256
    self.detail = detail
  }
}

public struct ReplyEvent: Codable, Sendable, Equatable {
  public let id: Int?
  public let ok: Bool
  public let error: String?
  public let extras: [String: JSONValue]

  public init(id: Int?, ok: Bool, error: String?, extras: [String: JSONValue] = [:]) {
    self.id = id
    self.ok = ok
    self.error = error
    self.extras = extras
  }

  private static let reserved: Set<String> = ["event", "t", "id", "ok", "error"]

  public init(from decoder: any Decoder) throws {
    let object = try decoder.singleValueContainer().decode([String: JSONValue].self)
    self.id = object["id"]?.intValue
    self.ok = object["ok"]?.boolValue ?? false
    self.error = object["error"]?.stringValue
    self.extras = object.filter { !ReplyEvent.reserved.contains($0.key) }
  }

  public func encode(to encoder: any Encoder) throws {
    var object = extras
    object["ok"] = .bool(ok)
    object["id"] = id.map { JSONValue.number(Double($0)) } ?? .null
    object["error"] = error.map { JSONValue.string($0) } ?? .null
    var container = encoder.singleValueContainer()
    try container.encode(object)
  }
}

// A JSON value of any shape, so reply extras survive decoding without a schema.
public enum JSONValue: Codable, Sendable, Equatable {
  case string(String)
  case number(Double)
  case bool(Bool)
  case null
  indirect case array([JSONValue])
  indirect case object([String: JSONValue])

  public var stringValue: String? {
    if case .string(let value) = self { return value }
    return nil
  }

  public var numberValue: Double? {
    if case .number(let value) = self { return value }
    return nil
  }

  public var intValue: Int? {
    if case .number(let value) = self { return Int(value) }
    return nil
  }

  public var boolValue: Bool? {
    if case .bool(let value) = self { return value }
    return nil
  }

  public var isNull: Bool {
    if case .null = self { return true }
    return false
  }

  public init(from decoder: any Decoder) throws {
    let container = try decoder.singleValueContainer()
    if container.decodeNil() {
      self = .null
    } else if let value = try? container.decode(Bool.self) {
      self = .bool(value)
    } else if let value = try? container.decode(Double.self) {
      self = .number(value)
    } else if let value = try? container.decode(String.self) {
      self = .string(value)
    } else if let value = try? container.decode([JSONValue].self) {
      self = .array(value)
    } else if let value = try? container.decode([String: JSONValue].self) {
      self = .object(value)
    } else {
      throw DecodingError.dataCorruptedError(in: container, debugDescription: "unsupported JSON value")
    }
  }

  public func encode(to encoder: any Encoder) throws {
    var container = encoder.singleValueContainer()
    switch self {
    case .string(let value): try container.encode(value)
    case .number(let value): try container.encode(value)
    case .bool(let value): try container.encode(value)
    case .null: try container.encodeNil()
    case .array(let value): try container.encode(value)
    case .object(let value): try container.encode(value)
    }
  }
}

/// mean, p50, p90, p99 and max of one measure over a benchmark, in ms.
public struct BenchmarkStats: Codable, Sendable, Equatable {
  public let mean: Double
  public let p50: Double
  public let p90: Double
  public let p99: Double
  public let max: Double

  public init(mean: Double, p50: Double, p90: Double, p99: Double, max: Double) {
    self.mean = mean
    self.p50 = p50
    self.p90 = p90
    self.p99 = p99
    self.max = max
  }

  public static let empty = BenchmarkStats(mean: 0, p50: 0, p90: 0, p99: 0, max: 0)
}

/// One window of a benchmark, with the device's thermal state as it closed.
public struct BenchmarkWindow: Codable, Sendable, Equatable {
  public let startSecond: Int
  public let frame: BenchmarkStats
  public let thermal: String

  public init(startSecond: Int, frame: BenchmarkStats, thermal: String) {
    self.startSecond = startSecond
    self.frame = frame
    self.thermal = thermal
  }
}

/// A finished benchmark: the loaded engine run at the comma's pace with
/// nothing on the link, so a phone's numbers can be read at home.
public struct BenchmarkReport: Codable, Sendable, Equatable {
  public let sha256: String
  public let device: String
  public let seconds: Double
  public let frames: Int
  /// The server's share of each frame: queues, model, output.
  public let frame: BenchmarkStats
  /// The model alone: the gpu_us the comma is told.
  public let accelerator: BenchmarkStats
  /// The history queues building the model's inputs, and the output read back.
  public let queues: BenchmarkStats
  public let output: BenchmarkStats
  /// How this build was compiled and what ran beside it, for comparing reports.
  public let build: String
  public let over35: Int
  public let over50: Int
  public let windows: [BenchmarkWindow]
  public let thermalAtStart: String
  public let thermalAtEnd: String
  public let cancelled: Bool

  public init(
    sha256: String, device: String, seconds: Double, frames: Int, frame: BenchmarkStats, accelerator: BenchmarkStats, queues: BenchmarkStats,
    output: BenchmarkStats, build: String, over35: Int, over50: Int, windows: [BenchmarkWindow], thermalAtStart: String, thermalAtEnd: String,
    cancelled: Bool
  ) {
    self.sha256 = sha256
    self.device = device
    self.seconds = seconds
    self.frames = frames
    self.frame = frame
    self.accelerator = accelerator
    self.queues = queues
    self.output = output
    self.build = build
    self.over35 = over35
    self.over50 = over50
    self.windows = windows
    self.thermalAtStart = thermalAtStart
    self.thermalAtEnd = thermalAtEnd
    self.cancelled = cancelled
  }

  /// The report as text, to paste into an issue or a note.
  public var text: String {
    func f(_ s: BenchmarkStats) -> String {
      String(format: "mean %.1f  p50 %.1f  p90 %.1f  p99 %.1f  max %.1f ms", s.mean, s.p50, s.p90, s.p99, s.max)
    }
    var lines = [
      "Jetlink benchmark, \(device)",
      "model \(sha256.prefix(16)), \(frames) frames at 20 Hz over \(Int(seconds)) s\(cancelled ? " (stopped early)" : "")",
      build,
      "frame        \(f(frame))",
      "accelerator  \(f(accelerator))",
      "queues       \(f(queues))",
      "output       \(f(output))",
      "over 35 ms: \(over35)   over 50 ms: \(over50)",
      "temperature: \(thermalAtStart) at start, \(thermalAtEnd) at end",
    ]
    if !windows.isEmpty {
      lines.append("by window:")
      for w in windows {
        lines.append(String(format: "  %4d s  mean %5.1f  p99 %5.1f  max %5.1f ms  ", w.startSecond, w.frame.mean, w.frame.p99, w.frame.max) + w.thermal)
      }
    }
    return lines.joined(separator: "\n")
  }
}

/// A benchmark's progress, and at its end the report. `state` is running,
/// done, cancelled or failed; `frame` is the running frame stats so far.
public struct BenchmarkEvent: Codable, Sendable, Equatable {
  public let state: String
  public let elapsed: Double
  public let total: Double
  public let frames: Int
  public let frame: BenchmarkStats?
  public let report: BenchmarkReport?
  public let detail: String

  public init(state: String, elapsed: Double, total: Double, frames: Int, frame: BenchmarkStats?, report: BenchmarkReport?, detail: String) {
    self.state = state
    self.elapsed = elapsed
    self.total = total
    self.frames = frames
    self.frame = frame
    self.report = report
    self.detail = detail
  }

  public var isFinished: Bool { state != "running" }
}

/// The comma asked the server to power its device off. The Swift server
/// answers no (a phone does not power itself off for the comma) and tells
/// the app, which tells the person.
public struct ShutdownRequestEvent: Codable, Sendable, Equatable {
  public let reason: String

  public init(reason: String) {
    self.reason = reason
  }
}

public enum ControlEvent: Sendable, Equatable {
  case hello(HelloEvent)
  case server(ServerEvent)
  case link(LinkEvent)
  case engine(EngineEvent)
  case stats(StatsEvent)
  case inventory(InventoryEvent)
  case catalog(CatalogEvent)
  case download(DownloadEvent)
  case importEvent(ImportEvent)
  case benchmark(BenchmarkEvent)
  case shutdownRequest(ShutdownRequestEvent)
  case reply(ReplyEvent)
  case unknown(name: String)

  private struct Envelope: Decodable {
    let event: String
  }

  public static func makeDecoder() -> JSONDecoder {
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    return decoder
  }

  public init(jsonLine: Data) throws {
    let decoder = ControlEvent.makeDecoder()
    let name = try decoder.decode(Envelope.self, from: jsonLine).event
    switch name {
    case "hello": self = .hello(try decoder.decode(HelloEvent.self, from: jsonLine))
    case "server": self = .server(try decoder.decode(ServerEvent.self, from: jsonLine))
    case "link": self = .link(try decoder.decode(LinkEvent.self, from: jsonLine))
    case "engine": self = .engine(try decoder.decode(EngineEvent.self, from: jsonLine))
    case "stats": self = .stats(try decoder.decode(StatsEvent.self, from: jsonLine))
    case "inventory": self = .inventory(try decoder.decode(InventoryEvent.self, from: jsonLine))
    case "catalog": self = .catalog(try decoder.decode(CatalogEvent.self, from: jsonLine))
    case "download": self = .download(try decoder.decode(DownloadEvent.self, from: jsonLine))
    case "import": self = .importEvent(try decoder.decode(ImportEvent.self, from: jsonLine))
    case "benchmark": self = .benchmark(try decoder.decode(BenchmarkEvent.self, from: jsonLine))
    case "shutdown_request": self = .shutdownRequest(try decoder.decode(ShutdownRequestEvent.self, from: jsonLine))
    case "reply": self = .reply(try decoder.decode(ReplyEvent.self, from: jsonLine))
    default: self = .unknown(name: name)
    }
  }

  public var replyEvent: ReplyEvent? {
    if case .reply(let reply) = self { return reply }
    return nil
  }
}

extension ControlEvent {
  /// The event's name on the wire: "hello", "import", "shutdown_request".
  public var name: String {
    switch self {
    case .hello: "hello"
    case .server: "server"
    case .link: "link"
    case .engine: "engine"
    case .stats: "stats"
    case .inventory: "inventory"
    case .catalog: "catalog"
    case .download: "download"
    case .importEvent: "import"
    case .benchmark: "benchmark"
    case .shutdownRequest: "shutdown_request"
    case .reply: "reply"
    case .unknown(let name): name
    }
  }

  /// The event as the Python server wrote it on the control channel: one
  /// JSON object with `event`, `t` (Unix seconds) and the payload's
  /// snake_case keys, then a newline. What the status page streams.
  public func jsonLine(at date: Date = Date()) -> Data {
    var object = payload()
    object["event"] = name
    object["t"] = date.timeIntervalSince1970
    var data = (try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys, .withoutEscapingSlashes])) ?? Data("{}".utf8)
    data.append(0x0A)
    return data
  }

  /// The payload alone, as JSONSerialization objects, with every absent
  /// optional written as null the way Python writes None: a page reading a
  /// key gets null, not undefined.
  public func payload() -> [String: Any] {
    switch self {
    case .hello(let event): ControlEvent.object(event)
    case .server(let event): ControlEvent.object(event)
    case .link(let event): ControlEvent.object(event)
    case .engine(let event): ControlEvent.object(event)
    case .stats(let event): ControlEvent.object(event)
    case .inventory(let event): ControlEvent.object(event)
    case .catalog(let event): ControlEvent.object(event)
    case .download(let event): ControlEvent.object(event)
    case .importEvent(let event): ControlEvent.object(event)
    case .benchmark(let event): ControlEvent.object(event)
    case .shutdownRequest(let event): ControlEvent.object(event)
    case .reply(let event): ControlEvent.object(event)
    case .unknown: [:]
    }
  }

  private static func object(_ value: some Encodable) -> [String: Any] {
    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    guard let data = try? encoder.encode(value), let object = try? JSONSerialization.jsonObject(with: data) else { return [:] }
    return withNulls(object, value) as? [String: Any] ?? [:]
  }

  /// JSONEncoder leaves a nil optional out, so each one the value holds is
  /// put back as null under the key the encoder would have used.
  private static func withNulls(_ encoded: Any, _ value: Any) -> Any {
    let mirror = Mirror(reflecting: value)
    switch (encoded, mirror.displayStyle) {
    case (var object as [String: Any], .struct?):
      for child in mirror.children {
        guard let label = child.label else { continue }
        let key = snakeCase(label)
        if let nested = object[key] {
          object[key] = withNulls(nested, child.value)
        } else if isNil(child.value) {
          object[key] = NSNull()
        }
      }
      return object
    case (let array as [Any], .collection?):
      return zip(array, mirror.children).map { withNulls($0, $1.value) }
    case (_, .optional?):
      return mirror.children.first.map { withNulls(encoded, $0.value) } ?? encoded
    default:
      return encoded
    }
  }

  private static func isNil(_ value: Any) -> Bool {
    let mirror = Mirror(reflecting: value)
    return mirror.displayStyle == .optional && mirror.children.isEmpty
  }

  /// `convertToSnakeCase` for the names these events use, none of which has
  /// two capitals in a row: "runtimeVersion" is "runtime_version".
  private static func snakeCase(_ name: String) -> String {
    var out = ""
    for character in name {
      if character.isUppercase {
        if !out.isEmpty { out.append("_") }
        out.append(contentsOf: character.lowercased())
      } else {
        out.append(character)
      }
    }
    return out
  }
}

public enum ControlCommand: Sendable, Equatable {
  case status
  case catalog(refresh: Bool)
  case download(ref: String?, sha256: String?)
  case cancelDownload(sha256: String)
  case importModel(path: String)
  case prepare(sha256: String, frameSkip: Int)
  case unload
  case forget(sha256: String, artifacts: Bool, model: Bool)
  case inventory
  case shutdown
  /// Run the loaded engine at the comma's pace for `seconds`, with no comma.
  case benchmark(seconds: Double)
  case cancelBenchmark

  public var name: String {
    switch self {
    case .status: return "status"
    case .catalog: return "catalog"
    case .download: return "download"
    case .cancelDownload: return "cancel_download"
    case .importModel: return "import"
    case .prepare: return "prepare"
    case .unload: return "unload"
    case .forget: return "forget"
    case .inventory: return "inventory"
    case .shutdown: return "shutdown"
    case .benchmark: return "benchmark"
    case .cancelBenchmark: return "cancel_benchmark"
    }
  }

  // Absent optional arguments are written as null, never left out, the way
  // section 4 of the contract describes the whole protocol.
  private var arguments: [String: Any] {
    switch self {
    case .status, .unload, .inventory, .shutdown, .cancelBenchmark:
      return [:]
    case .benchmark(let seconds):
      return ["seconds": seconds]
    case .catalog(let refresh):
      return ["refresh": refresh]
    case .download(let ref, let sha256):
      return ["ref": ref ?? NSNull(), "sha256": sha256 ?? NSNull()]
    case .cancelDownload(let sha256):
      return ["sha256": sha256]
    case .importModel(let path):
      return ["path": path]
    case .prepare(let sha256, let frameSkip):
      return ["sha256": sha256, "frame_skip": frameSkip]
    case .forget(let sha256, let artifacts, let model):
      return ["sha256": sha256, "artifacts": artifacts, "model": model]
    }
  }

  public func jsonLine(id: Int) -> Data {
    var object: [String: Any] = arguments
    object["id"] = id
    object["cmd"] = name
    guard var data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys, .withoutEscapingSlashes]) else {
      return Data("{}\n".utf8)
    }
    data.append(0x0A)
    return data
  }

  /// The command a control-protocol object names, `jsonLine(id:)` read back:
  /// what the Android app sends, as the server's command. Absent arguments
  /// take the Python server's defaults.
  public init(object: [String: Any]) throws {
    let name = object["cmd"] as? String ?? ""
    func text(_ key: String) throws -> String {
      guard let value = object[key] as? String, !value.isEmpty else { throw Invalid(description: "\(name) needs \(key)") }
      return value
    }
    func flag(_ key: String, _ fallback: Bool) -> Bool { (object[key] as? Bool) ?? fallback }
    func number(_ key: String) -> NSNumber? { object[key] as? NSNumber }
    switch name {
    case "status": self = .status
    case "catalog": self = .catalog(refresh: flag("refresh", false))
    case "download": self = .download(ref: object["ref"] as? String, sha256: object["sha256"] as? String)
    case "cancel_download": self = .cancelDownload(sha256: try text("sha256"))
    case "import": self = .importModel(path: try text("path"))
    case "prepare": self = .prepare(sha256: try text("sha256"), frameSkip: number("frame_skip")?.intValue ?? Pinned.defaultFrameSkip)
    case "unload": self = .unload
    case "forget": self = .forget(sha256: try text("sha256"), artifacts: flag("artifacts", true), model: flag("model", false))
    case "inventory": self = .inventory
    case "shutdown": self = .shutdown
    case "benchmark": self = .benchmark(seconds: number("seconds")?.doubleValue ?? 60)
    case "cancel_benchmark": self = .cancelBenchmark
    default: throw Invalid(description: "unknown command \(name)")
    }
  }

  /// A command object the server cannot run.
  public struct Invalid: Error, CustomStringConvertible {
    public let description: String
  }
}
