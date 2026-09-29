#if os(macOS) || os(Linux)
  import ArgumentParser
  import Foundation
  import JetlinkORT
  import JetlinkServer
  #if os(Linux)
    import JetlinkLinux
    import JetlinkTRT
  #endif

  /// What runs the model. `auto` is TensorRT where it loads, else onnxruntime.
  enum BackendName: String, CaseIterable, ExpressibleByArgument {
    case auto, trt, ort
  }

  /// --backend and --device, as every command that runs a model takes them.
  struct BackendArguments: ParsableArguments {
    @Option(help: "auto, trt or ort. auto takes TensorRT where it loads, else onnxruntime; a named one that cannot run here is an error.")
    var backend = BackendName.auto
    @Option(
      help: ArgumentHelp(
        "trt: a CUDA device index (0). ort: ane (default), ane-whole, coreml or cpu on a Mac; cpu on Linux.", valueName: "device"))
    var device: String?
    @Flag(help: "TensorRT: time each launch with CUDA events and log their spread every 1,200 frames.")
    var gpuTiming = false

    /// The options, with "auto" read as each backend's default device, as
    /// the Python server's --device took it.
    func options(keepAlive: Bool = true, keepCPUWarm: Bool = true) -> BackendOptions {
      BackendOptions(device: device == "auto" ? nil : device, keepAlive: keepAlive, keepCPUWarm: keepCPUWarm, gpuTiming: gpuTiming)
    }

    /// The backend asked for; with none, logs why and exits 1. `auto` logs
    /// why it passed over each one it did not take.
    func pick(keepAlive: Bool = true, keepCPUWarm: Bool = true) throws -> any EngineBackend {
      let log = ServerLog(category: "main")
      let backend: any EngineBackend
      do {
        backend = try options(keepAlive: keepAlive, keepCPUWarm: keepCPUWarm).pick(self.backend) { name, why in
          log.info("not using \(name.rawValue): \(why)")
        }
      } catch {
        log.error("\(error)")
        throw ExitCode.failure
      }
      if gpuTiming && backend.name != BackendName.trt.rawValue {
        log.warning("--gpu-timing times TensorRT's launches; \(backend.name) has no such timing")
      }
      return backend
    }
  }

  /// What the backends read from the command line.
  struct BackendOptions {
    /// trt: a CUDA device index. ort: ane, ane-whole, coreml or cpu on a Mac,
    /// cpu on Linux. nil for each one's default.
    var device: String?
    var keepAlive = true
    var keepCPUWarm = true
    var gpuTiming = false

    /// TensorRT, if it loads here and the GPU answers. A build without
    /// TensorRT's headers has the fake shim, which never loads.
    /// JETLINK_FAULT_CUDA_AFTER=N makes every frame after the first N fail
    /// as a sticky CUDA error does, for the fatal exit's acceptance run (H6).
    func trt() throws -> any EngineBackend {
      #if os(Linux)
        var index = 0
        if let device {
          guard let parsed = Int(device), parsed >= 0 else { throw HostError.invalid("--device \(device) is not a CUDA device index") }
          index = parsed
        }
        let faultAfter = ProcessInfo.processInfo.environment["JETLINK_FAULT_CUDA_AFTER"].flatMap { Int($0) }
        return TrtBackend(
          trt: try TensorRT(device: index), gpuTiming: gpuTiming, faultAfter: faultAfter, available: { Platform.memAvailableBytes() })
      #else
        throw HostError.invalid("TensorRT runs on Linux only")
      #endif
    }

    /// onnxruntime: CoreML on a Mac, its CPU provider on Linux, where the
    /// library is opened at run time and may be missing.
    func ort() throws -> any EngineBackend {
      let available = OrtProfile.available
      let name = device ?? available[0].rawValue
      guard let profile = OrtProfile(rawValue: name), available.contains(profile) else {
        throw HostError.invalid("onnxruntime has no device \(name) here: \(available.map(\.rawValue).joined(separator: ", "))")
      }
      #if !canImport(Metal)
        try OrtRuntime.load()
      #endif
      return OrtBackend(profile: profile, preparer: ONNXPreparer(), keepAlive: keepAlive, keepCPUWarm: keepCPUWarm)
    }

    /// `trt` or `ort`, made or refused.
    func make(_ name: BackendName) -> Result<any EngineBackend, any Error> {
      Result { name == .trt ? try trt() : try ort() }
    }

    /// The backend `name` asks for, or why there is none. `auto` tries
    /// TensorRT, then onnxruntime, and says why it passed over each one it
    /// did not take.
    func pick(_ name: BackendName, skipped: (BackendName, any Error) -> Void = { _, _ in }) throws -> any EngineBackend {
      guard name == .auto else { return try make(name).get() }
      var reasons: [String] = []
      for candidate in [BackendName.trt, .ort] {
        switch make(candidate) {
        case .success(let backend): return backend
        case .failure(let why):
          skipped(candidate, why)
          reasons.append("\(candidate.rawValue): \(why)")
        }
      }
      throw HostError.failed("no backend can run here (\(reasons.joined(separator: "; ")))")
    }
  }

  /// `jetlink-server backends`: what can run here and why the rest cannot.
  /// The installer asks it whether this machine has a usable GPU.
  struct ListBackends: ParsableCommand {
    static let configuration = CommandConfiguration(
      commandName: "backends",
      abstract: "List the backends and why each can or cannot run here.",
      discussion: "Exits 0 when one can: --backend trt tries TensorRT alone, so it asks for a GPU.")

    @OptionGroup var chosen: BackendArguments

    func run() throws {
      let options = chosen.options()
      var usable = false
      for name in chosen.backend == .auto ? [BackendName.trt, .ort] : [chosen.backend] {
        switch options.make(name) {
        case .success(let backend):
          print("\(name.rawValue): usable: \(name == .trt ? "TensorRT" : "onnxruntime") \(backend.runtimeVersion) on \(backend.deviceTag())")
          usable = true
        case .failure(let why):
          print("\(name.rawValue): not usable: \(why)")
        }
      }
      guard usable else { throw ExitCode.failure }
    }
  }
#endif
