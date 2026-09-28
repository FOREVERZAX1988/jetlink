import Android
import Foundation
import JetlinkKit
import JetlinkServer

// The Android app's way into the server: plain JNI functions on
// io.zoompilot.jetlink.server.Native, strings in and out.
//
// Kotlin draws and Swift decides. The app sends control commands in the
// control protocol's JSON (docs/control-protocol.md), and reads one snapshot
// of everything its screens show, built here from the same events and the
// same ModelRowBuilder the iPhone and Mac apps use. So the Android app holds
// no server logic of its own.
//
// Every call that waits (command, snapshot) is made from a Kotlin background
// thread. Nothing here calls back into Java, and nothing runs on the main
// actor, which a library loaded through JNI never drains.

// MARK: JNI strings

public typealias Env = UnsafeMutablePointer<JNIEnv?>

func string(_ env: Env, _ value: jstring?) -> String {
  guard let value, let chars = env.pointee!.pointee.GetStringUTFChars(env, value, nil) else { return "" }
  defer { env.pointee!.pointee.ReleaseStringUTFChars(env, value, chars) }
  return String(cString: chars)
}

/// JSON the app parses, as a Java string. JNI takes modified UTF-8, so
/// anything outside the Basic Multilingual Plane goes as a \u escape pair.
func jstring(_ env: Env, _ text: String) -> jstring? {
  var safe = ""
  safe.reserveCapacity(text.utf8.count)
  for scalar in text.unicodeScalars {
    if scalar.value > 0xFFFF {
      let v = scalar.value - 0x10000
      safe += String(format: "\\u%04x\\u%04x", 0xD800 + (v >> 10), 0xDC00 + (v & 0x3FF))
    } else if scalar.value == 0 {
      safe += "\\u0000"
    } else {
      safe.unicodeScalars.append(scalar)
    }
  }
  return safe.withCString { env.pointee!.pointee.NewStringUTF(env, $0) }
}

func json(_ object: Any) -> String {
  guard JSONSerialization.isValidJSONObject(object),
    let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys, .withoutEscapingSlashes])
  else { return "{}" }
  return String(decoding: data, as: UTF8.self)
}

func parse(_ text: String) -> [String: Any] {
  (try? JSONSerialization.jsonObject(with: Data(text.utf8))) as? [String: Any] ?? [:]
}

// MARK: the calls

/// `config` is JSON: cache (a directory), device ("htp", "htp-whole", "gpu",
/// "cpu"), keep_alive, keep_cpu_warm, listen, port, usb, preload, chip
/// (Build.SOC_MODEL) and native_library_dir. Returns nil, or why the server
/// could not start.
@_cdecl("Java_io_zoompilot_jetlink_server_Native_start")
public func nativeStart(_ env: Env, _ cls: jclass?, _ config: jstring?) -> jstring? {
  do {
    try Host.shared.start(parse(string(env, config)))
    return nil
  } catch {
    return jstring(env, String(describing: error))
  }
}

/// Stops the server and releases the engine.
@_cdecl("Java_io_zoompilot_jetlink_server_Native_stop")
public func nativeStop(_ env: Env, _ cls: jclass?) {
  Host.shared.stop()
}

/// Runs one control command, `{"cmd": "prepare", "sha256": ...}`, and returns
/// its reply, `{"ok": true, ...}`. Blocks until the server replies.
@_cdecl("Java_io_zoompilot_jetlink_server_Native_command")
public func nativeCommand(_ env: Env, _ cls: jclass?, _ command: jstring?) -> jstring? {
  jstring(env, json(Host.shared.command(parse(string(env, command)))))
}

/// The app's state once its version is past `after`, or after `timeoutMs`
/// either way.
@_cdecl("Java_io_zoompilot_jetlink_server_Native_snapshot")
public func nativeSnapshot(_ env: Env, _ cls: jclass?, _ after: jlong, _ timeoutMs: jint) -> jstring? {
  let server = Host.shared.running?.server
  let snapshot = Host.shared.state.snapshot(after: Int(after), timeout: Double(timeoutMs) / 1000, port: server?.port.map { Int($0) }) {
    server?.recentStats(window: AppSnapshot.recentWindow)
  }
  return jstring(env, json(snapshot))
}

/// Log lines numbered past `after`: `{"next": n, "lines": [...]}`.
@_cdecl("Java_io_zoompilot_jetlink_server_Native_logs")
public func nativeLogs(_ env: Env, _ cls: jclass?, _ after: jlong) -> jstring? {
  jstring(env, json(Host.shared.logs.lines(after: Int(after))))
}

/// The comma's gadget is open: `fd` from UsbDeviceConnection with the vendor
/// interface claimed, and its bulk endpoints' addresses. Nil or an error.
@_cdecl("Java_io_zoompilot_jetlink_server_Native_usbAttach")
public func nativeUSBAttach(_ env: Env, _ cls: jclass?, _ fd: jint, _ inEndpoint: jint, _ outEndpoint: jint) -> jstring? {
  do {
    try AndroidGadget.shared.attach(
      fd: fd, inEndpoint: UInt8(truncatingIfNeeded: inEndpoint), outEndpoint: UInt8(truncatingIfNeeded: outEndpoint))
    return nil
  } catch {
    return jstring(env, String(describing: error))
  }
}

/// The gadget is going. Returns once nothing uses the descriptor, so the app
/// can close the connection after.
@_cdecl("Java_io_zoompilot_jetlink_server_Native_usbDetach")
public func nativeUSBDetach(_ env: Env, _ cls: jclass?) {
  AndroidGadget.shared.detach()
}

/// "nominal", "fair", "serious" or "critical", from PowerManager.
@_cdecl("Java_io_zoompilot_jetlink_server_Native_reportThermal")
public func nativeReportThermal(_ env: Env, _ cls: jclass?, _ label: jstring?) {
  DeviceThermal.report(string(env, label))
}

/// What the comma logs about its accelerator with each reply: the phone's
/// temperatures and the like, as JSON.
@_cdecl("Java_io_zoompilot_jetlink_server_Native_reportTelemetry")
public func nativeReportTelemetry(_ env: Env, _ cls: jclass?, _ telemetry: jstring?) {
  Host.shared.telemetry.set(parse(string(env, telemetry)))
}

/// Versions and names the About screen shows: the runtime, the chip, the
/// devices this build can run on.
@_cdecl("Java_io_zoompilot_jetlink_server_Native_info")
public func nativeInfo(_ env: Env, _ cls: jclass?) -> jstring? {
  jstring(
    env,
    json([
      "onnxruntime": OrtRuntime.version,
      "chip": QNNBackend.chipName(),
      "devices": QNNBackend.Device.allCases.map(\.rawValue),
      "prepare_version": QNNBackend.prepareVersion,
    ] as [String: Any]))
}

// MARK: the server

/// The one server the app runs, and what it has said.
final class Host: @unchecked Sendable {
  static let shared = Host()

  let state = AppSnapshot()
  let logs = LogRing(capacity: 5000)
  let telemetry = Locked<[String: Any]>([:])
  private let lock = NSLock()
  private var embedded: EmbeddedServer?
  private var stream: LogStream?

  var running: EmbeddedServer? {
    lock.lock()
    defer { lock.unlock() }
    return embedded
  }

  func start(_ config: [String: Any]) throws {
    stop()
    if let dir = config["native_library_dir"] as? String, !dir.isEmpty {
      // Where the QNN provider finds the NPU's skel libraries: the app's own,
      // then the system's DSP directories.
      setenv("ADSP_LIBRARY_PATH", "\(dir);/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp", 1)
    }
    if let chip = config["chip"] as? String {
      QNNBackend.reportChip(chip)
    }
    guard let cache = config["cache"] as? String, !cache.isEmpty else {
      throw HostFailure("no cache directory")
    }
    let deviceName = config["device"] as? String ?? Server.defaultDevice.rawValue
    guard let device = Server.Device(rawValue: deviceName) else {
      throw HostFailure("unknown device \(deviceName)")
    }
    let configuration = Server.Configuration(
      port: UInt16(clamping: (config["port"] as? NSNumber)?.intValue ?? Int(Wire.defaultPort)),
      cacheRoot: URL(fileURLWithPath: cache, isDirectory: true),
      device: device,
      keepAlive: (config["keep_alive"] as? Bool) ?? true,
      keepCPUWarm: (config["keep_cpu_warm"] as? Bool) ?? false,
      preload: (config["preload"] as? Bool) ?? true,
      listen: (config["listen"] as? Bool) ?? true,
      usb: (config["usb"] as? Bool) ?? true)

    let logs = self.logs
    let stream = LogStream()
    Task.detached {
      for await line in stream.lines {
        logs.append(line)
      }
    }
    let server: EmbeddedServer
    do {
      server = try EmbeddedServer(configuration: configuration)
      let telemetry = self.telemetry
      server.server.telemetry = { telemetry.get() }
      state.reset()
      let events = server.events
      let state = self.state
      Task.detached {
        for await event in events {
          state.apply(event)
        }
      }
      try server.start()
    } catch {
      stream.finish()
      throw error
    }
    lock.lock()
    embedded = server
    self.stream = stream
    lock.unlock()
    state.serverStarted()
  }

  func stop() {
    lock.lock()
    let server = embedded
    let stream = self.stream
    embedded = nil
    self.stream = nil
    lock.unlock()
    server?.stop(releasingEngine: true)
    stream?.finish()
    if server != nil {
      state.serverStopped()
    }
  }

  func command(_ request: [String: Any]) -> [String: Any] {
    guard let server = running else {
      return ["ok": false, "error": "The server is not running."]
    }
    let command: ControlCommand
    do {
      command = try decode(request)
    } catch {
      return ["ok": false, "error": String(describing: error)]
    }
    let box = Locked<ReplyEvent?>(nil)
    let done = DispatchSemaphore(value: 0)
    Task.detached {
      box.set(await server.handle(command))
      done.signal()
    }
    done.wait()
    guard let reply = box.get() else { return ["ok": false, "error": "no reply"] }
    var out: [String: Any] = ["ok": reply.ok]
    if let error = reply.error { out["error"] = error }
    for (key, value) in reply.extras {
      out[key] = controlJSON(value)
    }
    return out
  }

  /// The control protocol's command object as the server's command.
  func decode(_ request: [String: Any]) throws -> ControlCommand {
    let name = request["cmd"] as? String ?? ""
    func text(_ key: String) throws -> String {
      guard let value = request[key] as? String, !value.isEmpty else { throw HostFailure("\(name) needs \(key)") }
      return value
    }
    func flag(_ key: String, _ fallback: Bool) -> Bool { (request[key] as? Bool) ?? fallback }
    switch name {
    case "status": return .status
    case "catalog": return .catalog(refresh: flag("refresh", false))
    case "download": return .download(ref: request["ref"] as? String, sha256: request["sha256"] as? String)
    case "cancel_download": return .cancelDownload(sha256: try text("sha256"))
    case "import": return .importModel(path: try text("path"))
    case "prepare":
      return .prepare(sha256: try text("sha256"), frameSkip: (request["frame_skip"] as? NSNumber)?.intValue ?? Pinned.defaultFrameSkip)
    case "unload": return .unload
    case "forget": return .forget(sha256: try text("sha256"), artifacts: flag("artifacts", true), model: flag("model", false))
    case "inventory": return .inventory
    case "benchmark": return .benchmark(seconds: (request["seconds"] as? NSNumber)?.doubleValue ?? 60)
    case "cancel_benchmark": return .cancelBenchmark
    default: throw HostFailure("unknown command \(name)")
    }
  }
}

struct HostFailure: Error, CustomStringConvertible {
  let description: String
  init(_ description: String) { self.description = description }
}

/// A value behind a lock.
final class Locked<Value>: @unchecked Sendable {
  private let lock = NSLock()
  private var value: Value

  init(_ value: Value) { self.value = value }

  func get() -> Value {
    lock.lock()
    defer { lock.unlock() }
    return value
  }

  func set(_ newValue: Value) {
    lock.lock()
    value = newValue
    lock.unlock()
  }
}

/// The last `capacity` log lines, numbered so the Logs screen asks only for
/// what it has not shown.
final class LogRing: @unchecked Sendable {
  private let lock = NSLock()
  private var kept: [String] = []
  private var total = 0
  let capacity: Int

  init(capacity: Int) { self.capacity = capacity }

  func append(_ line: String) {
    lock.lock()
    kept.append(line)
    if kept.count > capacity {
      kept.removeFirst(kept.count - capacity)
    }
    total += 1
    lock.unlock()
    AndroidLog.write(line)
  }

  /// Lines numbered after `after` (0 for all kept), and the number to ask
  /// after next time.
  func lines(after: Int) -> [String: Any] {
    lock.lock()
    defer { lock.unlock() }
    let oldest = total - kept.count
    let start = min(max(after, oldest) - oldest, kept.count)
    return ["next": total, "lines": Array(kept[start...])]
  }
}

/// The server's log in logcat too, where `adb logcat -s jetlink` finds it.
enum AndroidLog {
  static func write(_ line: String) {
    // ANDROID_LOG_ERROR, WARN, INFO
    let priority: Int32 = line.contains(" ERROR ") ? 6 : line.contains(" WARNING ") ? 5 : 4
    _ = __android_log_write(priority, "jetlink", line)
  }
}
