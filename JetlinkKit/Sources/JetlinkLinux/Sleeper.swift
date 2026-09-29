#if os(Linux)
  import Foundation
  import Glibc
  import JetlinkServer

  /// Suspends the box once the comma has been gone long enough (Python's
  /// `jetlink/server/sleep.py`).
  ///
  /// On an always-on supply a Jetson idles at about 7 W, a flat battery over
  /// a long park. Deep suspend keeps the engine resident: about 6 s back,
  /// against about 65 s for a cold boot and a plan reload. USB is the wake
  /// source and both edges wake it, so this is a loop, not a command from the
  /// comma: ignition-off pulls the gadget and wakes the box, and no gadget for
  /// `after` seconds means sleep again. Writing `mem` to /sys/power/state
  /// blocks until resume, so the USB loop carries on where it stopped with
  /// the engine, the CUDA context and the gadget's descriptor intact.
  ///
  /// `jetlink caffeinate` holds the box awake with a shared flock on
  /// /run/jetlink-awake.lock; a held lock puts off the suspend until it goes.
  public final class Sleeper: @unchecked Sendable {
    public static let retryMin: TimeInterval = 10
    public static let retryMax: TimeInterval = 300
    /// The USB wake is not guaranteed: a sleeping Jetson once answered a
    /// bind with a bus reset and no enumeration through four connect cycles
    /// and needed its button, which in the car is a drive on the small model.
    /// An RTC alarm does not depend on the path that failed, and costs under
    /// a tenth of a watt.
    public static let wakeBackstop: TimeInterval = 1800
    /// Outside any systemd RuntimeDirectory, so a restart of the server never
    /// unlinks it under a holder.
    public static let awakeLock = "/run/jetlink-awake.lock"

    /// CLOCK_MONOTONIC for the idle count, which stands still while asleep,
    /// and CLOCK_BOOTTIME, which does not, to time the sleep.
    public struct Clock: Sendable {
      public var monotonic: @Sendable () -> TimeInterval
      public var boottime: @Sendable () -> TimeInterval

      public init(monotonic: @escaping @Sendable () -> TimeInterval, boottime: @escaping @Sendable () -> TimeInterval) {
        self.monotonic = monotonic
        self.boottime = boottime
      }

      public static let system = Clock(monotonic: { now(CLOCK_MONOTONIC) }, boottime: { now(CLOCK_BOOTTIME) })

      private static func now(_ clock: clockid_t) -> TimeInterval {
        var time = timespec()
        clock_gettime(clock, &time)
        return TimeInterval(time.tv_sec) + TimeInterval(time.tv_nsec) / 1e9
      }
    }

    public let after: TimeInterval
    let root: HostRoot
    let lockPath: String
    let backstop: TimeInterval
    let clock: Clock
    /// Every write to sysfs: `mem` to /sys/power/state returns once the box
    /// is back. A test's stand-in for the kernel.
    let write: @Sendable (String, String) throws(KernelError) -> Void
    let log: LinuxLog

    private let lock = NSLock()
    private(set) var enabled = true
    private var lastSeen: TimeInterval
    private var retryAt: TimeInterval = 0
    private var backoff = Sleeper.retryMin
    private var held = false
    private(set) var slept = 0
    private(set) var failed = 0

    public convenience init(after: TimeInterval) {
      self.init(after: after, root: .system)
    }

    init(
      after: TimeInterval, root: HostRoot, lockPath: String? = nil, backstop: TimeInterval = Sleeper.wakeBackstop, clock: Clock = .system,
      write: @escaping @Sendable (String, String) throws(KernelError) -> Void = Sysfs.write, log: @escaping LinuxLog = serverLog("sleep")
    ) {
      self.after = after
      self.root = root
      self.lockPath = lockPath ?? root.path(Sleeper.awakeLock)
      self.backstop = backstop
      self.clock = clock
      self.write = write
      self.log = log
      lastSeen = clock.monotonic()
    }

    /// `ServerHooks.gadgetIdle`: any sign of the comma restarts the count;
    /// only an absent gadget may end in a suspend. True when the box slept.
    public func handle(_ event: GadgetIdleEvent) -> Bool {
      switch event {
      case .present, .connected, .disconnected:
        touch()
        return false
      case .absent:
        return idle()
      }
    }

    public func touch() {
      lock.withLock {
        lastSeen = clock.monotonic()
        backoff = Sleeper.retryMin
      }
    }

    /// Called while no gadget is present. Suspends once the count has run
    /// out and returns true when the box slept, back awake.
    public func idle() -> Bool {
      let due = lock.withLock {
        let now = clock.monotonic()
        return enabled && now - lastSeen >= after && now >= retryAt
      }
      guard due else { return false }
      if heldAwake() {
        let first = lock.withLock {
          defer { held = true }
          return !held
        }
        if first { log(.info, "held awake by jetlink caffeinate") }
        return false
      }
      let released = lock.withLock {
        defer { held = false }
        return held
      }
      if released {
        // The count starts again from the release, not from the comma.
        log(.info, "no longer held awake; suspending after \(Int(after)) s more without a gadget")
        touch()
        return false
      }
      let ok = suspend()
      lock.withLock {
        let now = clock.monotonic()
        if ok {
          // Woken by an edge: give whatever caused it the whole count to show up.
          lastSeen = now
          backoff = Sleeper.retryMin
        } else {
          retryAt = now + backoff
          backoff = min(backoff * 2, Sleeper.retryMax)
        }
      }
      return ok
    }

    /// Creates the hold-awake lock for `jetlink caffeinate`, world-readable
    /// so a holder needs no sudo. Throws when it cannot (not root).
    public static func createAwakeLock(_ path: String = Sleeper.awakeLock) throws(KernelError) {
      let fd = open(path, O_RDONLY | O_CREAT | O_CLOEXEC, 0o644)
      guard fd >= 0 else { throw KernelError(what: path, errno: errno) }
      // The umask may have taken bits off.
      fchmod(fd, 0o644)
      close(fd)
    }

    /// Whether a `jetlink caffeinate` holds the lock. A holder that died let
    /// go with its descriptor, so a stale one never counts.
    func heldAwake() -> Bool {
      let fd = open(lockPath, O_RDONLY | O_CLOEXEC)
      guard fd >= 0 else { return false }
      defer { close(fd) }
      if flock(fd, LOCK_EX | LOCK_NB) == 0 {
        flock(fd, LOCK_UN)
        return false
      }
      return errno == EWOULDBLOCK
    }

    // MARK: the suspend

    private func suspend() -> Bool {
      guard selectDeep() else {
        lock.withLock { enabled = false }
        return false
      }
      let power = root.path("/sys/power")
      let successes = Sysfs.readInt("\(power)/suspend_stats/success") ?? -1
      let start = clock.boottime()
      log(.info, "no gadget for \(Int(after)) s, suspending")
      armHubWakeup()
      let armed = armBackstop()
      do throws(KernelError) {
        try write("\(power)/state", "mem")
      } catch {
        disarmBackstop(armed)
        lock.withLock { failed += 1 }
        if [EACCES, EPERM, EROFS, ENOENT].contains(error.errno) {
          // Configuration, not weather: nothing will change by the next try.
          log(.error, "cannot write \(power)/state (\(error)); sleep disabled")
          lock.withLock { enabled = false }
        } else {
          // EBUSY is the freezer giving up, EINVAL a mode the platform
          // refused: both are worth another try later.
          log(.warning, "suspend failed: \(error) (\(failure()))")
        }
        return false
      }
      disarmBackstop(armed)
      let asleep = clock.boottime() - start
      if successes >= 0 && (Sysfs.readInt("\(power)/suspend_stats/success") ?? -1) <= successes {
        // A clean return with the counter unmoved: a wake edge landed during
        // the freeze and the box never left.
        log(.warning, "suspend returned after \(String(format: "%.1f", asleep)) s without sleeping (\(failure()))")
        lock.withLock { failed += 1 }
        return false
      }
      lock.withLock { slept += 1 }
      log(.info, "resumed after \(Int(asleep.rounded())) s asleep")
      return true
    }

    /// Deep suspend: s2idle keeps the CPUs in idle states and saves nothing
    /// worth having. False when deep is not to be had.
    private func selectDeep() -> Bool {
      let path = root.path("/sys/power/mem_sleep")
      // No mem_sleep at all: "mem" means whatever the platform does.
      guard let modes = Sysfs.read(path), !modes.isEmpty else { return true }
      if modes.contains("[deep]") { return true }
      guard modes.split(separator: " ").contains("deep") else {
        log(.error, "deep suspend is not available (mem_sleep: \(modes)), not sleeping")
        return false
      }
      do {
        try write(path, "deep")
      } catch {
        log(.error, "could not select deep suspend: \(error)")
        return false
      }
      return true
    }

    /// Arms remote wakeup on every hub, and says so loudly when it cannot.
    /// The comma hangs off the carrier's Realtek hub, which has to signal a
    /// connect up before the root hub hears about it and which ships
    /// disarmed: disarmed, a sleeping Jetson answered a bind with a bus reset
    /// and needed its button; armed, it resumed 4 s after the bind. Which hub
    /// carries the gadget depends on the negotiated speed, so all of them.
    @discardableResult
    func armHubWakeup() -> [String] {
      let devices = "/sys/bus/usb/devices"
      var disarmed: [String] = []
      for device in root.list(devices) {
        // Interfaces ("2-1:1.0") have no bDeviceClass and are skipped here.
        guard Sysfs.read(root.path("\(devices)/\(device)/bDeviceClass")) == "09" else { continue }
        let wakeup = root.path("\(devices)/\(device)/power/wakeup")
        guard Sysfs.read(wakeup) == "disabled" else { continue }
        do {
          try write(wakeup, "enabled\n")
        } catch {
          disarmed.append(device)
        }
      }
      if !disarmed.isEmpty {
        log(
          .error,
          "hub(s) \(disarmed.joined(separator: ", ")) could not be armed for remote wakeup: the comma presenting its gadget may not wake this box"
        )
      }
      return disarmed
    }

    /// An RTC alarm `backstop` seconds out, against the RTC's own count:
    /// this box boots unset and never sees NTP in the car, so only both sides
    /// coming from the same counter matters.
    private func armBackstop() -> Bool {
      guard backstop > 0 else { return false }
      let rtc = root.path("/sys/class/rtc/rtc0")
      do {
        guard let now = Sysfs.readInt("\(rtc)/since_epoch") else {
          throw KernelError(what: "\(rtc)/since_epoch", errno: ENODATA)
        }
        // A stale alarm blocks setting a new one.
        try write("\(rtc)/wakealarm", "0\n")
        try write("\(rtc)/wakealarm", "\(now + Int(backstop))\n")
        return true
      } catch {
        // The USB edge is still the wake source; this was only the backstop.
        log(.warning, "could not arm the \(Int(backstop)) s wake backstop: \(error)")
        return false
      }
    }

    private func disarmBackstop(_ armed: Bool) {
      guard armed else { return }
      // It has either fired or fires once and is spent: a failure is fine.
      try? write(root.path("/sys/class/rtc/rtc0/wakealarm"), "0\n")
    }

    private func failure() -> String {
      let stats = root.path("/sys/power/suspend_stats")
      let step = Sysfs.read("\(stats)/last_failed_step").flatMap { $0.isEmpty ? nil : $0 } ?? "?"
      let device = Sysfs.read("\(stats)/last_failed_dev") ?? ""
      return "last failed step \(step)" + (device.isEmpty ? "" : " in \(device)")
    }
  }
#endif
