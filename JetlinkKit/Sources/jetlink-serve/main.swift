import Foundation
import JetlinkKit
import JetlinkServer

// The Swift server on a Mac, for benches: the same code the iPhone runs, over
// TCP, against scripts/bench_link.py or a comma on the LAN.
//
//   jetlink-serve --cache DIR [options]
//
// Events print as JSON lines on stdout, as the control channel would send
// them. `--help` lists the options.

let usage = """
  usage: jetlink-serve --cache DIR [options]

  The jetlink server in Swift, the one the iPhone app runs, on a Mac for
  benches: serves a comma or scripts/bench_link.py over TCP and prints the
  control channel's events as JSON lines.

  options:
    --cache DIR         where models and built engines live (required)
    --port PORT         TCP port to listen on (default 5599)
    --device ane|coreml|cpu
                        ane: the vision trunk on the Neural Engine, the rest on
                        the GPU (default); coreml: the whole graph on the GPU;
                        cpu: onnxruntime's CPU provider, for tests
    --dial HOST[:PORT]  also dial this end and serve the connection, as the
                        phone dials the comma over a USB network link; the
                        listener stays open beside it
    --no-keepalive      do not keep the GPU clocked up between frames
    --no-cpu-keepwarm   do not keep a CPU core busy between frames while a
                        Neural Engine session runs
    --no-preload        do not load the engine loaded last before a comma asks
    --help              this text
  """

struct Options {
  var cache: URL?
  var port: UInt16 = 5599
  var device: CoreMLBackend.Device = .ane
  var keepAlive = true
  var keepCPUWarm = true
  var preload = true
  var dial: DialTarget?
}

func parse() -> Options {
  var options = Options()
  var args = CommandLine.arguments.dropFirst()
  func value(_ flag: String) -> String {
    guard let next = args.popFirst() else {
      FileHandle.standardError.write(Data("\(flag) needs a value\n".utf8))
      exit(2)
    }
    return next
  }
  while let arg = args.popFirst() {
    switch arg {
    case "--cache": options.cache = URL(fileURLWithPath: value(arg), isDirectory: true)
    case "--port": options.port = UInt16(value(arg)) ?? 5599
    case "--device": options.device = CoreMLBackend.Device(rawValue: value(arg)) ?? .ane
    case "--no-keepalive": options.keepAlive = false
    case "--no-cpu-keepwarm": options.keepCPUWarm = false
    case "--no-preload": options.preload = false
    case "--help", "-h":
      print(usage)
      exit(0)
    case "--dial":
      let text = value(arg)
      guard let target = DialTarget(text) else {
        FileHandle.standardError.write(Data("--dial wants HOST or HOST:PORT, not \(text)\n".utf8))
        exit(2)
      }
      options.dial = target
    default:
      FileHandle.standardError.write(Data("unknown argument \(arg); --help lists the options\n".utf8))
      exit(2)
    }
  }
  return options
}

func line(_ event: String, _ payload: [String: Any]) {
  var object = payload
  object["event"] = event
  object["t"] = Date().timeIntervalSince1970
  if let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys, .withoutEscapingSlashes]),
    let text = String(data: data, encoding: .utf8)
  {
    print(text)
    fflush(stdout)
  }
}

func encode<T: Encodable>(_ value: T) -> [String: Any] {
  let encoder = JSONEncoder()
  encoder.keyEncodingStrategy = .convertToSnakeCase
  guard let data = try? encoder.encode(value),
    let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
  else { return [:] }
  return object
}

let options = parse()
guard let cache = options.cache else {
  FileHandle.standardError.write(Data((usage + "\n").utf8))
  exit(2)
}

do {
  let server = try Server(
    configuration: Server.Configuration(
      port: options.port, cacheRoot: cache, device: options.device, keepAlive: options.keepAlive, keepCPUWarm: options.keepCPUWarm,
      preload: options.preload, dial: options.dial),
    preparer: ONNXPreparer())
  server.host.subscribe { event in
    switch event {
    case .progress(let stage, let frac, let msg): line("progress", ["stage": stage, "frac": frac, "msg": msg])
    case .engine(let engine): line("engine", encode(engine))
    case .link(let link): line("link", encode(link))
    case .stats(let stats): line("stats", encode(stats))
    case .benchmark(let benchmark): line("benchmark", encode(benchmark))
    }
  }
  line("server", server.backend.describe())
  try server.start()
  signal(SIGINT, SIG_IGN)
  let interrupt = DispatchSource.makeSignalSource(signal: SIGINT, queue: .main)
  interrupt.setEventHandler {
    server.shutdown()
    exit(0)
  }
  interrupt.resume()
  dispatchMain()
} catch {
  FileHandle.standardError.write(Data("jetlink-serve: \(error)\n".utf8))
  exit(1)
}
