#if os(macOS) || os(Linux)
  import ArgumentParser
  import Foundation
  import JetlinkORT
  import JetlinkServer
  #if os(Linux)
    import JetlinkTRT
  #endif

  /// What runs the model. `auto` is TensorRT where it loads, else onnxruntime.
  enum BackendName: String, CaseIterable, ExpressibleByArgument {
    case auto, trt, ort
  }

  struct BackendUnusable: Error, CustomStringConvertible {
    let description: String

    init(_ description: String) {
      self.description = description
    }
  }

  /// --backend and --device, as every command that runs a model takes them.
  struct BackendArguments: ParsableArguments {
    @Option(help: "auto, trt or ort. auto takes TensorRT where it loads, else onnxruntime; a named one that cannot run here is an error.")
    var backend = BackendName.auto
    @Option(
      help: ArgumentHelp(
        "trt: a CUDA device index (0). ort: ane (default), ane-whole, coreml or cpu on a Mac; cpu on Linux.", valueName: "device"))
    var device: String?

    /// The options, with "auto" read as each backend's default device, as
    /// the Python server's --device took it.
    func options(keepAlive: Bool = true, keepCPUWarm: Bool = true) -> BackendOptions {
      BackendOptions(device: device == "auto" ? nil : device, keepAlive: keepAlive, keepCPUWarm: keepCPUWarm)
    }

    /// The backend asked for; with none, logs why and exits 1. `auto` logs
    /// why it passed over each one it did not take.
    func pick(keepAlive: Bool = true, keepCPUWarm: Bool = true) throws -> any EngineBackend {
      let log = ServerLog(category: "main")
      do {
        return try options(keepAlive: keepAlive, keepCPUWarm: keepCPUWarm).pick(backend) { name, why in
          log.info("not using \(name.rawValue): \(why)")
        }
      } catch {
        log.error("\(error)")
        throw ExitCode.failure
      }
    }
  }

  /// What the backends read from the command line.
  struct BackendOptions {
    /// trt: a CUDA device index. ort: ane, ane-whole, coreml or cpu on a Mac,
    /// cpu on Linux. nil for each one's default.
    var device: String?
    var keepAlive = true
    var keepCPUWarm = true

    /// TensorRT, if it loads here and this build has its backend.
    func trt() throws -> any EngineBackend {
      #if os(Linux)
        var index = 0
        if let device {
          guard let parsed = Int(device), parsed >= 0 else { throw BackendUnusable("--device \(device) is not a CUDA device index") }
          index = parsed
        }
        let found: String
        do {
          found = try TensorRT.probe(device: index)
        } catch {
          throw BackendUnusable(String(describing: error))
        }
        throw BackendUnusable("\(found) loads, but this build has no TensorRT backend yet")
      #else
        throw BackendUnusable("TensorRT runs on Linux only")
      #endif
    }

    /// onnxruntime: CoreML on a Mac, its CPU provider on Linux, where the
    /// library is opened at run time and may be missing.
    func ort() throws -> any EngineBackend {
      #if canImport(Metal)
        let name = device ?? CoreMLBackend.Device.ane.rawValue
        guard let unit = CoreMLBackend.Device(rawValue: name) else {
          throw BackendUnusable("onnxruntime has no device \(name) here: ane, ane-whole, coreml or cpu")
        }
        return CoreMLBackend(device: unit, preparer: ONNXPreparer(), keepAlive: keepAlive, keepCPUWarm: keepCPUWarm)
      #else
        let name = device ?? QNNBackend.Device.cpu.rawValue
        guard name == QNNBackend.Device.cpu.rawValue else { throw BackendUnusable("onnxruntime runs on the CPU here, not \(name)") }
        do {
          try OrtRuntime.load()
        } catch {
          throw BackendUnusable(String(describing: error))
        }
        return QNNBackend(device: .cpu, preparer: ONNXPreparer(), keepAlive: keepAlive, keepCPUWarm: keepCPUWarm)
      #endif
    }

    /// Each backend in the order `auto` tries them, made or refused.
    func candidates() -> [(name: BackendName, backend: Result<any EngineBackend, BackendUnusable>)] {
      [(.trt, attempt(trt)), (.ort, attempt(ort))]
    }

    /// The backend `name` asks for, or why there is none. `auto` says why it
    /// passed over each one it did not take.
    func pick(_ name: BackendName, skipped: (BackendName, BackendUnusable) -> Void = { _, _ in }) throws -> any EngineBackend {
      switch name {
      case .trt: return try trt()
      case .ort: return try ort()
      case .auto:
        var reasons: [String] = []
        for (candidate, made) in candidates() {
          switch made {
          case .success(let backend): return backend
          case .failure(let why):
            skipped(candidate, why)
            reasons.append("\(candidate.rawValue): \(why)")
          }
        }
        throw BackendUnusable("no backend can run here (\(reasons.joined(separator: "; ")))")
      }
    }

    private func attempt(_ make: () throws -> any EngineBackend) -> Result<any EngineBackend, BackendUnusable> {
      do {
        return .success(try make())
      } catch let why as BackendUnusable {
        return .failure(why)
      } catch {
        return .failure(BackendUnusable(String(describing: error)))
      }
    }
  }

  /// `jetlink-server backends`: what can run here and why the rest cannot.
  /// The installer asks it whether this machine has a usable GPU.
  struct ListBackends: ParsableCommand {
    static let configuration = CommandConfiguration(
      commandName: "backends",
      abstract: "List the backends and why each can or cannot run here.",
      discussion: "Exits 0 when the backend --backend names can run: auto's pick by default, so --backend trt asks for TensorRT and a GPU.")

    @OptionGroup var chosen: BackendArguments

    func run() throws {
      let found = chosen.options().candidates()
      for (name, made) in found {
        switch made {
        case .success(let backend):
          let runtime = backend.name == BackendName.trt.rawValue ? "TensorRT" : "onnxruntime"
          print("\(name.rawValue): usable: \(runtime) \(backend.runtimeVersion) on \(backend.deviceTag())")
        case .failure(let why):
          print("\(name.rawValue): not usable: \(why)")
        }
      }
      let usable = found.first { (name, made) in
        guard case .success = made else { return false }
        return chosen.backend == .auto || chosen.backend == name
      }
      guard usable != nil else { throw ExitCode.failure }
    }
  }
#endif
