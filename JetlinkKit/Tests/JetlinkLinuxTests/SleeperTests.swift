#if os(Linux)
  import Foundation
  import Glibc
  import JetlinkServer
  import Testing

  @testable import JetlinkLinux

  /// The sleeper on a copy of the Jetson's sysfs and a clock the test moves.
  /// The kernel is the writer: `mem` to /sys/power/state sleeps (the success
  /// count rises), or fails with the errno the test chooses.
  final class Kernel: @unchecked Sendable {
    let tree = Tree.jetsonCopy()
    let monotonic = Dial(1000.0)
    let lines = Lines()
    /// Each write, as (path under the tree, text).
    let writes = Dial<[(String, String)]>([])
    /// errno for a write to a path ending so.
    let failing = Dial<[String: Int32]>([:])
    /// Whether a suspend sleeps, or returns at once without sleeping.
    let sleeps = Dial(true)

    /// Made at once, so the idle count starts with the kernel.
    private(set) var sleeper: Sleeper!

    init() {
      tree.makeDirectory("/run")
      sleeper = Sleeper(
        after: 120, root: tree.root, lockPath: tree.path("/run/jetlink-awake.lock"), monotonic: { [monotonic] in monotonic.value },
        write: { [unowned self] path, text throws(KernelError) in try kernelWrite(path, text) }, log: lines.log)
    }

    private func kernelWrite(_ path: String, _ text: String) throws(KernelError) {
      let relative = String(path.dropFirst(tree.url.path.count))
      writes.value.append((relative, text))
      if let error = failing.value.first(where: { relative.hasSuffix($0.key) })?.value {
        throw KernelError(what: path, errno: error)
      }
      if relative == "/sys/power/state" {
        if sleeps.value {
          let count = Int(tree.read("/sys/power/suspend_stats/success")!.trimmingCharacters(in: .whitespacesAndNewlines))!
          tree.write("/sys/power/suspend_stats/success", "\(count + 1)\n")
        }
        return
      }
      try Sysfs.write(path, text)
    }

    var suspends: Int { writes.value.filter { $0.0 == "/sys/power/state" }.count }

    func wait(_ seconds: TimeInterval) {
      monotonic.value += seconds
    }
  }

  @Suite("Sleeper")
  struct SleeperTests {
    @Test("No gadget for sleep_after seconds: deep suspend, and back")
    func sleeps() {
      let kernel = Kernel()
      #expect(!kernel.sleeper.handle(.absent))
      kernel.wait(119)
      #expect(!kernel.sleeper.handle(.absent))
      kernel.wait(1)
      #expect(kernel.sleeper.handle(.absent))
      #expect(kernel.suspends == 1)
      #expect(kernel.lines.has(.info, "no gadget for 120 s, suspending"))
      #expect(kernel.lines.has(.info, "resumed after"))
      // Deep is selected already ("s2idle [deep]"), and every hub is armed.
      #expect(!kernel.writes.value.contains { $0.0 == "/sys/power/mem_sleep" })
      #expect(!kernel.writes.value.contains { $0.0.hasSuffix("/power/wakeup") })
      // Awake again: whatever woke it gets the whole count to show up.
      kernel.wait(119)
      #expect(!kernel.sleeper.handle(.absent))
      kernel.wait(1)
      #expect(kernel.sleeper.handle(.absent))
      #expect(kernel.suspends == 2)
    }

    @Test("The gadget, a connection or its end restarts the count")
    func touches() {
      for event in [GadgetIdleEvent.present, .connected, .disconnected] {
        let kernel = Kernel()
        kernel.wait(100)
        #expect(!kernel.sleeper.handle(event))
        kernel.wait(100)
        #expect(!kernel.sleeper.handle(.absent))
        kernel.wait(20)
        #expect(kernel.sleeper.handle(.absent))
      }
    }

    @Test("An RTC alarm half an hour out on the RTC's own count, cleared first, disarmed after")
    func backstop() throws {
      let kernel = Kernel()
      kernel.wait(120)
      #expect(kernel.sleeper.idle())
      let alarms = kernel.writes.value.filter { $0.0 == "/sys/class/rtc/rtc0/wakealarm" }.map(\.1)
      // since_epoch was 1790625382
      #expect(alarms == ["0\n", "1790627182\n", "0\n"])
      let order = kernel.writes.value.map(\.0)
      let suspend = try #require(order.firstIndex(of: "/sys/power/state"))
      #expect(suspend > (try #require(order.firstIndex(of: "/sys/class/rtc/rtc0/wakealarm"))))
    }

    @Test("No RTC: a warning, and the suspend goes ahead on the USB wake alone")
    func noRTC() {
      let kernel = Kernel()
      kernel.tree.remove("/sys/class/rtc")
      kernel.wait(120)
      #expect(kernel.sleeper.idle())
      #expect(kernel.lines.has(.warning, "could not arm the 1800 s wake backstop"))
    }

    @Test("A hub shipped disarmed is armed before the suspend; one that will not arm is said loudly")
    func hubs() {
      let kernel = Kernel()
      kernel.tree.write("/sys/bus/usb/devices/2-1/power/wakeup", "disabled\n")
      kernel.tree.write("/sys/bus/usb/devices/usb1/power/wakeup", "disabled\n")
      kernel.failing.value = ["/usb1/power/wakeup": EACCES]
      kernel.wait(120)
      #expect(kernel.sleeper.idle())
      #expect(kernel.tree.read("/sys/bus/usb/devices/2-1/power/wakeup") == "enabled\n")
      // 1-3 is Bluetooth (class e0), disabled and left alone.
      #expect(kernel.tree.read("/sys/bus/usb/devices/1-3/power/wakeup") == "disabled\n")
      #expect(kernel.lines.has(.error, "hub(s) usb1 could not be armed for remote wakeup"))
    }

    @Test("s2idle only: never sleeps; deep offered but not selected: selects it")
    func deep() {
      let kernel = Kernel()
      kernel.tree.write("/sys/power/mem_sleep", "[s2idle]\n")
      kernel.wait(120)
      #expect(!kernel.sleeper.idle())
      #expect(!kernel.sleeper.enabled)
      kernel.wait(1000)
      #expect(!kernel.sleeper.idle())
      #expect(kernel.suspends == 0)
      #expect(kernel.lines.has(.error, "deep suspend is not available (mem_sleep: [s2idle])"))

      let offered = Kernel()
      offered.tree.write("/sys/power/mem_sleep", "[s2idle] deep\n")
      offered.wait(120)
      #expect(offered.sleeper.idle())
      #expect(offered.tree.read("/sys/power/mem_sleep") == "deep")

      let refused = Kernel()
      refused.tree.write("/sys/power/mem_sleep", "[s2idle] deep\n")
      refused.failing.value = ["/sys/power/mem_sleep": EINVAL]
      refused.wait(120)
      #expect(!refused.sleeper.idle())
      #expect(!refused.sleeper.enabled)

      let unknown = Kernel()
      unknown.tree.remove("/sys/power/mem_sleep")
      unknown.wait(120)
      #expect(unknown.sleeper.idle())
    }

    @Test("EBUSY, EINVAL and the rest back off 10 s, doubling to 300 s", arguments: [EBUSY, EINVAL, EIO, EAGAIN])
    func backsOff(errno: Int32) {
      let kernel = Kernel()
      kernel.failing.value = ["/sys/power/state": errno]
      kernel.wait(120)
      var attempts = 0
      for backoff in [10.0, 20, 40, 80, 160, 300, 300] {
        #expect(!kernel.sleeper.idle())
        attempts += 1
        #expect(kernel.suspends == attempts)
        kernel.wait(backoff - 1)
        #expect(!kernel.sleeper.idle())
        #expect(kernel.suspends == attempts)
        kernel.wait(1)
      }
      #expect(kernel.sleeper.enabled)
      #expect(kernel.lines.has(.warning, "suspend failed"))
      // Unlike a failure, the comma resets the backoff; the retry time stands.
      kernel.sleeper.touch()
      kernel.failing.value = [:]
      kernel.wait(120)
      #expect(kernel.sleeper.idle())
    }

    @Test("EACCES, EPERM, EROFS and ENOENT turn sleeping off for good", arguments: [EACCES, EPERM, EROFS, ENOENT])
    func disables(errno: Int32) {
      let kernel = Kernel()
      kernel.failing.value = ["/sys/power/state": errno]
      kernel.wait(120)
      #expect(!kernel.sleeper.idle())
      #expect(!kernel.sleeper.enabled)
      kernel.failing.value = [:]
      kernel.wait(10_000)
      #expect(!kernel.sleeper.idle())
      #expect(kernel.suspends == 1)
      #expect(kernel.lines.has(.error, "sleep disabled"))
      // The alarm set for the failed attempt is taken back.
      #expect(kernel.writes.value.filter { $0.0 == "/sys/class/rtc/rtc0/wakealarm" }.last?.1 == "0\n")
    }

    @Test("A write that returns with the success count unmoved never slept, and backs off")
    func wokeDuringFreeze() {
      let kernel = Kernel()
      kernel.tree.write("/sys/power/suspend_stats/last_failed_step", "suspend\n")
      kernel.tree.write("/sys/power/suspend_stats/last_failed_dev", "3610000.usb\n")
      kernel.sleeps.value = false
      kernel.wait(120)
      #expect(!kernel.sleeper.idle())
      #expect(kernel.lines.has(.warning, "without sleeping (last failed step suspend in 3610000.usb)"))
      kernel.wait(9)
      #expect(!kernel.sleeper.idle())
      kernel.sleeps.value = true
      kernel.wait(1)
      #expect(kernel.sleeper.idle())
    }
  }

  @Suite("Hold awake")
  struct HoldAwakeTests {
    /// A `jetlink caffeinate`: a shared flock on its own read-only open.
    final class Holder {
      let fd: Int32

      init(_ path: String) {
        fd = open(path, O_RDONLY | O_CLOEXEC)
        precondition(fd >= 0 && flock(fd, LOCK_SH) == 0)
      }

      func release() {
        close(fd)
      }
    }

    func kernel() -> Kernel {
      let kernel = Kernel()
      try! Sleeper.createAwakeLock(kernel.tree.path("/run/jetlink-awake.lock"))
      return kernel
    }

    @Test("Held: no suspend, said once, looked at again each poll")
    func held() {
      let kernel = kernel()
      let holder = Holder(kernel.tree.path("/run/jetlink-awake.lock"))
      defer { holder.release() }
      kernel.wait(120)
      for _ in 0..<5 {
        #expect(!kernel.sleeper.handle(.absent))
        kernel.wait(0.5)
      }
      #expect(kernel.suspends == 0)
      #expect(kernel.lines.count("held awake by jetlink caffeinate") == 1)
    }

    @Test("Released: the count starts again from the release")
    func released() {
      let kernel = kernel()
      let holder = Holder(kernel.tree.path("/run/jetlink-awake.lock"))
      kernel.wait(500)
      #expect(!kernel.sleeper.idle())
      holder.release()
      #expect(!kernel.sleeper.idle())
      #expect(kernel.lines.count("no longer held awake") == 1)
      kernel.wait(119)
      #expect(!kernel.sleeper.idle())
      kernel.wait(1)
      #expect(kernel.sleeper.idle())
      #expect(kernel.suspends == 1)
    }

    @Test("A holder that died let go with its descriptor: no hold")
    func staleHolder() {
      let kernel = kernel()
      Holder(kernel.tree.path("/run/jetlink-awake.lock")).release()
      kernel.wait(120)
      #expect(kernel.sleeper.idle())
      #expect(kernel.lines.count("held awake") == 0)
    }

    @Test("The daemon makes the lock world-readable whatever the umask, and keeps one that is there")
    func created() throws {
      let tree = Tree()
      let path = tree.path("/jetlink-awake.lock")
      let old = umask(0o077)
      defer { umask(old) }
      try Sleeper.createAwakeLock(path)
      var info = stat()
      stat(path, &info)
      #expect(info.st_mode & 0o777 == 0o644)
      let holder = Holder(path)
      defer { holder.release() }
      try Sleeper.createAwakeLock(path)
      #expect(FileManager.default.fileExists(atPath: path))
    }
  }
#endif
