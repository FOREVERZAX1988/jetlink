#if os(Linux)
  import Foundation
  import JetlinkServer

  /// A Jetson's or a Linux PC's side of the server: what jetlink-server passes
  /// it besides the backend and the gadget, `SysfsGadget`.
  public enum LinuxHost {
    /// The server's hooks on this machine: `telemetry`'s readings, the
    /// sleeper, the poweroff. `sleepAfter` is what was asked for; the hooks
    /// say what the host will do: 0 when it cannot suspend. `gadget` hears
    /// when sessions start and end, for the link's power management.
    public static func hooks(sleepAfter: Double, poweroff: Bool, telemetry: GPUTelemetry?, gadget: SysfsGadget? = nil) -> ServerHooks {
      hooks(sleepAfter: sleepAfter, poweroff: poweroff, telemetry: telemetry, gadget: gadget, root: .system)
    }

    static func hooks(
      sleepAfter: Double, poweroff: Bool, telemetry: GPUTelemetry?, gadget: SysfsGadget? = nil, root: HostRoot,
      log: @escaping LinuxLog = serverLog("linux")
    ) -> ServerHooks {
      var hooks = ServerHooks()
      if let telemetry {
        let sampler = TelemetrySampler(source: telemetry.read)
        hooks.telemetry = { sampler.read() }
      }

      // Made whether or not this server sleeps, so `jetlink caffeinate`
      // always has something to hold.
      let lockPath = root.path(Sleeper.awakeLock)
      var lockError: KernelError?
      do throws(KernelError) {
        try Sleeper.createAwakeLock(lockPath)
      } catch {
        lockError = error
      }
      var sleeper: Sleeper?
      if sleepAfter > 0 {
        if Platform.canSuspend(root) {
          sleeper = Sleeper(after: sleepAfter, root: root, lockPath: lockPath)
          hooks.sleepAfter = sleepAfter
          log(.info, "will suspend after \(Int(sleepAfter)) s without a gadget")
          if let lockError { log(.warning, "jetlink caffeinate cannot hold this box awake: \(lockError)") }
        } else {
          log(.warning, "--sleep-after needs /sys/power/state, which this host has not got: not suspending")
        }
      }
      if sleeper != nil || gadget != nil {
        hooks.gadgetIdle = { [sleeper] event in
          gadget?.hear(event)
          return sleeper?.handle(event) ?? false
        }
      }

      hooks.shutdown = PowerOff.hook(enabled: poweroff)
      return hooks
    }

    /// Tegra sysfs on a Jetson, else NVML for CUDA device `gpu` where the
    /// driver has it, else none, which the comma is told as `{}` rather than
    /// zeros.
    public static func telemetry(gpu: Int) -> GPUTelemetry? {
      telemetry(tegra: Platform.isTegra(), root: .system, gpu: gpu, nvml: NvmlTelemetry.open, log: serverLog("linux"))
    }

    static func telemetry(
      tegra: Bool, root: HostRoot, gpu: Int, nvml: (Int) -> Result<NvmlTelemetry, NvmlUnavailable>, log: LinuxLog
    ) -> GPUTelemetry? {
      if tegra {
        let telemetry = TegraTelemetry(root: root)
        log(.info, "telemetry from Tegra sysfs")
        return GPUTelemetry(read: { telemetry.read() }, name: nil, tegra: true)
      }
      switch nvml(gpu) {
      case .success(let telemetry):
        log(.info, "telemetry from NVML on GPU \(gpu) (\(telemetry.name ?? "unnamed"))")
        return GPUTelemetry(read: { telemetry.read() }, name: telemetry.name, tegra: false)
      case .failure(let why):
        log(.info, "no telemetry on this host: \(why)")
        return nil
      }
    }
  }

  /// The GPU's readings in the keys the comma logs, from the one source this
  /// host has: the telemetry hook and the status page's hardware panel read
  /// the same one.
  public struct GPUTelemetry: Sendable {
    public let read: @Sendable () -> [String: Any]
    /// NVML's name for the GPU; nil on a Jetson, whose device tree names it.
    public let name: String?
    public let tegra: Bool
  }
#endif
