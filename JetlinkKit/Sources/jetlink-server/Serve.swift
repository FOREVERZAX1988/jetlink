#if os(macOS) || os(Linux)
  import ArgumentParser
  import Dispatch
  import Foundation
  import JetlinkKit
  import JetlinkServer
  import JetlinkStatusPage
  #if os(Linux)
    import JetlinkLinux
    import JetlinkLog
  #endif

  enum LogLevel: String, CaseIterable, ExpressibleByArgument {
    case debug, info, warning, error
  }

  /// --cache, which every command that touches the cache takes.
  struct CacheArguments: ParsableArguments {
    @Option(help: "Where models and built engines live. Default: $JETLINK_CACHE, else /mnt/data/jetlink on a Jetson, else the user's cache directory.")
    var cache: String?

    var root: URL { cache.map { URL(fileURLWithPath: $0, isDirectory: true) } ?? defaultCache() }
  }

  /// `jetlink-server serve`, the default: serves the comma until SIGINT or
  /// SIGTERM, then exits 0.
  struct Serve: ParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Serve the comma (the default).")

    @OptionGroup var chosen: BackendArguments
    @Flag(help: "Be the USB host for the comma's gadget. No TCP listener then, unless --listen too.")
    var usb = false
    @Flag(help: "Listen on TCP, as without --usb.")
    var listen = false
    @Option(help: "The address to listen on.")
    var host = "0.0.0.0"
    @Option(help: "The TCP port to listen on.")
    var port = Wire.defaultPort
    @Option(help: ArgumentHelp("Also dial this end and serve it, as the phone dials the comma over a USB network link.", valueName: "host[:port]"))
    var dial: String?
    @OptionGroup var cache: CacheArguments
    @Option(help: "Suspend after this many seconds with no gadget (Linux, with --usb); 0 never.")
    var sleepAfter = 0.0
    @Option(help: "Serve the read-only status page on this port; 0 is off.")
    var statusPort = 0
    @Flag(name: .customLong("no-preload"), help: "Do not load the engine loaded last before a comma asks.")
    var noPreload = false
    @Flag(name: .customLong("no-keepalive"), help: "Do not keep the GPU clocked up between frames (a Mac's onnxruntime).")
    var noKeepAlive = false
    @Flag(name: .customLong("no-cpu-keepwarm"), help: "Do not keep a CPU core busy between Neural Engine frames (a Mac's onnxruntime).")
    var noCPUKeepWarm = false
    @Option(help: "debug, info, warning or error.")
    var logLevel = LogLevel.info

    func validate() throws {
      if let dial, DialTarget(dial) == nil {
        throw ValidationError("--dial wants HOST or HOST:PORT, not \(dial)")
      }
      guard sleepAfter >= 0 else { throw ValidationError("--sleep-after cannot be negative") }
      // A TCP listener never sees the comma go, so nothing would say when to sleep.
      guard sleepAfter == 0 || usb else { throw ValidationError("--sleep-after needs --usb") }
      guard (0...65535).contains(statusPort) else { throw ValidationError("--status-port \(statusPort) is not a port") }
    }

    func run() throws {
      holdStopSignals()
      setUpLogging(logLevel)
      let log = ServerLog(category: "main")
      let backend = try chosen.pick(keepAlive: !noKeepAlive, keepCPUWarm: !noCPUKeepWarm)
      let root = cache.root

      var hooks = ServerHooks()
      var gadget: (any GadgetSource)?
      #if os(macOS)
        gadget = USBGadget()
      #elseif os(Linux)
        // The Linux host's gadget, telemetry, sleeper and poweroff plug in here.
        hooks = LinuxHost.hooks(cache: root, sleepAfter: sleepAfter)
        gadget = LinuxHost.gadget()
      #endif
      if sleepAfter > 0 && hooks.sleepAfter != sleepAfter {
        log.warning("this build does not suspend this host: --sleep-after \(sleepAfter) is ignored, and the comma is told it never sleeps")
      }
      hooks.fatal = exitOnFatal

      let server: Server
      do {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        server = try Server(
          configuration: Server.Configuration(
            host: host, port: port, cacheRoot: root, preload: !noPreload, dial: dial.flatMap { DialTarget($0) }, listen: listen || !usb, usb: usb),
          backend: backend, gadget: gadget, hooks: hooks)
        log.info("backend \(backend.name) \(backend.runtimeVersion) on \(backend.deviceTag()), cache \(root.path)")
        try server.start()
      } catch {
        log.error("cannot serve: \(error)")
        throw ExitCode.failure
      }
      if statusPort > 0 {
        // The status page starts here, on its own threads, once it serves.
        // Without its page it stays off: the comma matters more than a page.
        do {
          _ = try StatusPage.page()
          log.warning("--status-port \(statusPort): this build does not serve the status page yet")
        } catch {
          log.warning("\(error)")
        }
      }
      stopOnSignals { signal in
        log.info("stopping on \(signal)")
        // The status page goes after the server, once it serves.
        shutDown([("the server", server.shutdown)], log: log)
      }
    }
  }

  /// Takes down what serves, in the order given, and says so: the comma's
  /// server and its engine first, so a frame in flight is answered or cut
  /// before anything else goes, then the status page, which shows the
  /// server stopping until the end.
  func shutDown(_ steps: [(name: String, stop: () -> Void)], log: ServerLog) {
    for step in steps {
      step.stop()
      log.info("stopped \(step.name)")
    }
  }

  /// Log lines on standard error, from `threshold` up: journald adds the time
  /// under systemd. Linux's loggers write there already; a Mac's go to the
  /// unified log, so they are copied out.
  func setUpLogging(_ threshold: LogLevel) {
    #if os(Linux)
      Logger.threshold =
        switch threshold {
        case .debug: .debug
        case .info: .info
        case .warning: .warning
        case .error: .error
        }
    #else
      let rank: [Log.Level: Int] = [.info: 1, .warning: 2, .error: 3]
      let least = [LogLevel.debug: 0, .info: 1, .warning: 2, .error: 3][threshold]!
      Log.sink = { level, category, message in
        guard rank[level]! >= least else { return }
        FileHandle.standardError.write(Data((EmbeddedServer.logLine(level, category, message) + "\n").utf8))
      }
    #endif
  }

  /// Exits 3 after an engine error nothing recovers from: only a new process
  /// gets a working device back, and systemd starts one, which preloads the
  /// engine.
  @Sendable func exitOnFatal(_ error: any Error) {
    ServerLog(category: "main").error("exiting on an engine error nothing recovers from: \(error)")
    exit(3)
  }

  let stopSignals = [(SIGINT, "SIGINT"), (SIGTERM, "SIGTERM")]

  /// Keeps SIGINT and SIGTERM pending for `stopOnSignals` rather than
  /// delivered: on Linux the dispatch source reads them from a signalfd, which
  /// only sees a signal no thread took, and a thread that has it unblocked
  /// takes and drops it. Called before any thread starts, so all inherit it.
  func holdStopSignals() {
    #if os(Linux)
      var held = sigset_t()
      sigemptyset(&held)
      for (number, _) in stopSignals {
        sigaddset(&held, number)
      }
      pthread_sigmask(SIG_BLOCK, &held, nil)
    #endif
  }

  /// Runs `stop` on the main queue at SIGINT or SIGTERM, then exits 0. Never
  /// returns.
  func stopOnSignals(_ stop: @escaping @Sendable (String) -> Void) -> Never {
    let sources = stopSignals.map { number, name in
      // Handled by the source alone; a shell that starts this in the
      // background hands it SIGINT ignored, which the source still hears.
      signal(number, SIG_IGN)
      let source = DispatchSource.makeSignalSource(signal: number, queue: .main)
      source.setEventHandler {
        stop(name)
        exit(0)
      }
      source.resume()
      return source
    }
    withExtendedLifetime(sources) { dispatchMain() }
  }

  /// $JETLINK_CACHE, else the Jetson's data partition, else the user's cache
  /// directory, as the Python server chose.
  func defaultCache(environment: [String: String] = ProcessInfo.processInfo.environment, tegra: () -> Bool = isTegra) -> URL {
    if let named = environment["JETLINK_CACHE"], !named.isEmpty {
      return URL(fileURLWithPath: named, isDirectory: true)
    }
    let jetson = URL(fileURLWithPath: "/mnt/data/jetlink", isDirectory: true)
    var isDirectory: ObjCBool = false
    if FileManager.default.fileExists(atPath: jetson.path, isDirectory: &isDirectory) && isDirectory.boolValue || tegra() {
      return jetson
    }
    let home = FileManager.default.homeDirectoryForCurrentUser
    #if os(macOS)
      return home.appending(path: "Library/Caches/jetlink", directoryHint: .isDirectory)
    #else
      let base = environment["XDG_CACHE_HOME"].flatMap { $0.isEmpty ? nil : URL(fileURLWithPath: $0, isDirectory: true) }
      return (base ?? home.appending(path: ".cache", directoryHint: .isDirectory)).appending(path: "jetlink", directoryHint: .isDirectory)
    #endif
  }

  /// Any one of these says Tegra, as the Python server looked.
  func isTegra() -> Bool {
    #if os(Linux)
      for path in ["/sys/firmware/devicetree/base/compatible", "/proc/device-tree/compatible"] {
        if let data = FileManager.default.contents(atPath: path), String(decoding: data, as: UTF8.self).lowercased().contains("tegra") {
          return true
        }
      }
      return ["/etc/nv_tegra_release", "/sys/devices/platform/bus@0/17000000.gpu"].contains { FileManager.default.fileExists(atPath: $0) }
    #else
      return false
    #endif
  }
#endif
