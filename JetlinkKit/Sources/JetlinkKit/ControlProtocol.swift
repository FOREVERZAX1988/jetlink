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

  public init(state: LinkState, detail: String, peer: String?) {
    self.state = state
    self.detail = detail
    self.peer = peer
  }

  public static let waiting = LinkEvent(state: .waiting, detail: "", peer: nil)
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

  public struct Gpu: Codable, Sendable, Equatable {
    public let mean: Double

    public init(mean: Double) {
      self.mean = mean
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

  public let frames: Int
  public let fps: Double
  public let totalMs: Total
  public let gpuMs: Gpu
  public let slow: Int
  public let windowS: Double
  /// Absent from a server older than the frame budget view.
  public var stagesMs: Stages? = nil
  /// From a frame's arrival to its reply leaving: `totalMs` plus the send.
  public var servedMs: Total? = nil

  public init(frames: Int, fps: Double, totalMs: Total, gpuMs: Gpu, slow: Int, windowS: Double, stagesMs: Stages? = nil, servedMs: Total? = nil) {
    self.frames = frames
    self.fps = fps
    self.totalMs = totalMs
    self.gpuMs = gpuMs
    self.slow = slow
    self.windowS = windowS
    self.stagesMs = stagesMs
    self.servedMs = servedMs
  }

  /// The stages, or for an older server the model run and everything else.
  public var stages: Stages {
    stagesMs ?? Stages(queue: 0, gpu: gpuMs.mean, other: max(0, totalMs.mean - gpuMs.mean), send: 0)
  }

  public var served: Total {
    servedMs ?? totalMs
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

  public init(sha256: String, key: String, path: String, bytes: Int64, backend: String, runtimeVersion: String?, device: String, builtAt: String?, buildSeconds: Double?, checkpoint: String?, current: Bool) {
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
    case "reply": self = .reply(try decoder.decode(ReplyEvent.self, from: jsonLine))
    default: self = .unknown(name: name)
    }
  }

  public var replyEvent: ReplyEvent? {
    if case .reply(let reply) = self { return reply }
    return nil
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
    }
  }

  // Absent optional arguments are written as null, never left out, the way
  // section 4 of the contract describes the whole protocol.
  private var arguments: [String: Any] {
    switch self {
    case .status, .unload, .inventory, .shutdown:
      return [:]
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
}
