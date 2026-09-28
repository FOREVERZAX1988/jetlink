#if os(Linux)
  import Foundation
  import JetlinkServer

  /// A Jetson's or a Linux PC's side of the server: what jetlink-server passes
  /// it besides the backend. The comma's gadget through sysfs, and behind
  /// `ServerHooks` the telemetry, the sleeper and the poweroff.
  public enum LinuxHost {
    /// The server's hooks on this machine. `sleepAfter` is what was asked
    /// for; the hooks say what the host will do: 0 when it cannot suspend.
    /// `gpu` is the CUDA device index NVML reports on (a PC's).
    public static func hooks(cache: URL, sleepAfter: Double, gpu: Int = 0) -> ServerHooks {
      hooks(cache: cache, sleepAfter: sleepAfter, gpu: gpu, root: .system)
    }

    static func hooks(
      cache: URL, sleepAfter: Double, gpu: Int, root: HostRoot, nvml: (Int) -> Result<NvmlTelemetry, NvmlUnavailable> = NvmlTelemetry.open,
      log: @escaping LinuxLog = serverLog("linux")
    ) -> ServerHooks {
      var hooks = ServerHooks()
      let tegra = Platform.isTegra(root)
      if let source = telemetrySource(tegra: tegra, root: root, gpu: gpu, nvml: nvml, log: log) {
        let sampler = TelemetrySampler(source: source)
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
      if sleepAfter > 0 {
        if Platform.canSuspend(root) {
          let sleeper = Sleeper(after: sleepAfter, root: root, lockPath: lockPath)
          hooks.sleepAfter = sleepAfter
          hooks.gadgetIdle = { sleeper.handle($0) }
          log(.info, "will suspend after \(Int(sleepAfter)) s without a gadget")
          if let lockError { log(.warning, "jetlink caffeinate cannot hold this box awake: \(lockError)") }
        } else {
          log(.warning, "--sleep-after needs /sys/power/state, which this host has not got: not suspending")
        }
      }

      hooks.shutdown = PowerOff.hook(cache: cache, tegra: tegra)
      return hooks
    }

    /// Tegra sysfs on a Jetson, else NVML where the driver has it, else none,
    /// which the comma is told as `{}` rather than zeros.
    static func telemetrySource(
      tegra: Bool, root: HostRoot, gpu: Int, nvml: (Int) -> Result<NvmlTelemetry, NvmlUnavailable>, log: LinuxLog
    ) -> TelemetrySampler.Source? {
      if tegra {
        let telemetry = TegraTelemetry(root: root)
        log(.info, "telemetry from Tegra sysfs")
        return { telemetry.read() }
      }
      switch nvml(gpu) {
      case .success(let telemetry):
        log(.info, "telemetry from NVML on GPU \(gpu) (\(telemetry.name ?? "unnamed"))")
        return { telemetry.read() }
      case .failure(let why):
        log(.info, "no telemetry on this host: \(why)")
        return nil
      }
    }

    /// The comma's gadget, found through sysfs and claimed through usbfs.
    public static func gadget() -> (any GadgetSource)? {
      SysfsGadget()
    }
  }
#endif
