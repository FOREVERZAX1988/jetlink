#if os(Linux)
  import Foundation
  import JetlinkServer

  /// A Jetson's or a Linux PC's side of the server: what jetlink-server passes
  /// it besides the backend. The comma's gadget found through sysfs, and the
  /// telemetry, sleeper and poweroff behind `ServerHooks`, belong here; until
  /// they are, the server listens and dials only, and its hooks are an app's.
  public enum LinuxHost {
    /// The server's hooks on this machine. `sleepAfter` is what was asked
    /// for; the hooks say what the host will do.
    public static func hooks(cache: URL, sleepAfter: Double) -> ServerHooks {
      ServerHooks()
    }

    /// The comma's gadget, once this host can find one on the bus.
    public static func gadget() -> (any GadgetSource)? {
      nil
    }
  }
#endif
