#if os(Linux)
  import Foundation
  import Glibc

  /// Powering the box off on the comma's say-so (D9). hardwared shuts the
  /// comma down below 11.8 V or after 30 hours parked, and a Jetson on an
  /// always-on feed draws even asleep, so it goes too. Off is off: only a DC
  /// cycle or the button brings it back, which is why only a Jetson does it.
  public enum PowerOff {
    /// In the cache, it keeps the box up: the installer writes it on a PC and
    /// when the person declined poweroff, and the tests before every server.
    public static let dryRunName = "poweroff-dry-run"

    /// `ServerHooks.shutdown` on Linux. Every request is answered ok, as the
    /// Python server did, so the comma sees what it always saw; once the
    /// reply is written the box syncs, then powers off if it is a Jetson and
    /// no dry-run file says otherwise.
    public static func hook(cache: URL, tegra: Bool) -> @Sendable (String) -> (@Sendable () -> Void)? {
      hook(cache: cache, tegra: tegra, powerOff: systemctlPowerOff, log: serverLog("power"))
    }

    static func hook(cache: URL, tegra: Bool, powerOff: @escaping @Sendable (@escaping LinuxLog) -> Void, log: @escaping LinuxLog)
      -> @Sendable (String) -> (@Sendable () -> Void)?
    {
      let dryRun = cache.appending(path: dryRunName).path
      return { _ in
        {
          sync()
          if FileManager.default.fileExists(atPath: dryRun) {
            log(.warning, "dry run: staying up (\(dryRun) exists)")
          } else if !tegra {
            log(.warning, "not a Jetson: staying up")
          } else {
            log(.warning, "powering off")
            powerOff(log)
          }
        }
      }
    }

    /// `systemctl poweroff`, found on PATH.
    static func systemctlPowerOff(_ log: @escaping LinuxLog) {
      guard let systemctl = Platform.which("systemctl", path: ProcessInfo.processInfo.environment["PATH"]) else {
        log(.error, "cannot power off: no systemctl on PATH")
        return
      }
      do {
        let pid = try spawn(systemctl, ["poweroff"])
        // Reaped off the session thread: systemd may stop this process
        // before systemctl returns.
        Thread.detachNewThread {
          let status = exitStatus(of: pid)
          if status != 0 { log(.error, "systemctl poweroff exited with status \(status)") }
        }
      } catch {
        log(.error, "cannot run \(systemctl) poweroff: \(error)")
      }
    }

    /// Starts `path` with `arguments` and this process's environment. The
    /// daemon blocks SIGINT and SIGTERM on every thread and ignores SIGPIPE,
    /// and a child inherits both through exec: the child gets an empty mask
    /// and those signals' defaults back.
    static func spawn(_ path: String, _ arguments: [String]) throws(KernelError) -> pid_t {
      var attributes = posix_spawnattr_t()
      posix_spawnattr_init(&attributes)
      defer { posix_spawnattr_destroy(&attributes) }
      var empty = sigset_t()
      sigemptyset(&empty)
      var defaults = sigset_t()
      sigemptyset(&defaults)
      for signal in [SIGINT, SIGTERM, SIGPIPE, SIGHUP] {
        sigaddset(&defaults, signal)
      }
      posix_spawnattr_setsigmask(&attributes, &empty)
      posix_spawnattr_setsigdefault(&attributes, &defaults)
      posix_spawnattr_setflags(&attributes, Int16(POSIX_SPAWN_SETSIGMASK | POSIX_SPAWN_SETSIGDEF))
      let argv = ([path] + arguments).map { strdup($0) } + [nil]
      defer { argv.forEach { free($0) } }
      var pid: pid_t = 0
      let error = posix_spawn(&pid, path, nil, &attributes, argv, environ)
      guard error == 0 else { throw KernelError(what: path, errno: error) }
      return pid
    }

    /// The child's exit status, or -1 when it did not exit normally.
    static func exitStatus(of pid: pid_t) -> Int32 {
      var status: Int32 = 0
      while waitpid(pid, &status, 0) < 0 {
        guard errno == EINTR else { return -1 }
      }
      // WIFEXITED and WEXITSTATUS, which are macros.
      return status & 0x7f == 0 ? (status >> 8) & 0xff : -1
    }
  }
#endif
