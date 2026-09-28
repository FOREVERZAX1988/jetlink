#if os(Linux)
  import Foundation
  import JetlinkServer
  import Testing

  @testable import JetlinkLinux

  @Suite("Linux host")
  struct LinuxHostTests {
    func hooks(_ tree: Tree, sleepAfter: Double, lines: Lines = Lines()) -> ServerHooks {
      LinuxHost.hooks(
        cache: tree.url, sleepAfter: sleepAfter, gpu: 0, root: tree.root, nvml: { _ in .failure(NvmlUnavailable(reason: "no driver")) },
        log: lines.log)
    }

    @Test("A Jetson suspends when asked to, and the hello says so")
    func sleeps() {
      let tree = Tree.jetsonCopy()
      tree.makeDirectory("/run")
      let lines = Lines()
      let hooks = hooks(tree, sleepAfter: 900, lines: lines)
      #expect(hooks.sleepAfter == 900)
      #expect(hooks.gadgetIdle?(.present) == false)
      #expect(lines.has(.info, "will suspend after 900 s without a gadget"))
      #expect(FileManager.default.fileExists(atPath: tree.path(Sleeper.awakeLock)))
    }

    @Test("Without /sys/power/state, or with --sleep-after 0, the hello says it never sleeps")
    func neverSleeps() {
      let tree = Tree()
      let lines = Lines()
      let hooks = hooks(tree, sleepAfter: 900, lines: lines)
      #expect(hooks.sleepAfter == 0)
      #expect(hooks.gadgetIdle == nil)
      #expect(lines.has(.warning, "needs /sys/power/state"))
      #expect(self.hooks(Tree.jetsonCopy(), sleepAfter: 0).sleepAfter == 0)
    }

    @Test("Every shutdown is accepted, so the comma hears ok")
    func shutdown() {
      let tree = Tree()
      tree.write("/\(PowerOff.dryRunName)", "")
      #expect(hooks(tree, sleepAfter: 0).shutdown?("car battery") != nil)
    }

    @Test("A Jetson's telemetry comes from its sysfs, a host with none sends {}")
    func telemetry() {
      let tree = Tree.jetsonCopy()
      let jetson = hooks(tree, sleepAfter: 0)
      #expect(eventually { jetson.telemetry()["supply_mv"] as? Int == 4952 })
      withExtendedLifetime(tree) {}
      let none = hooks(Tree(), sleepAfter: 0)
      Thread.sleep(forTimeInterval: 0.05)
      #expect(none.telemetry().isEmpty)
    }

    @Test("The gadget is the sysfs one")
    func gadget() {
      #expect(LinuxHost.gadget() is SysfsGadget)
    }
  }

  @Suite("Platform")
  struct PlatformTests {
    @Test("The bench Jetson is a Tegra; any one sign says so; an empty host is not")
    func tegra() {
      #expect(Platform.isTegra(jetson))
      #expect(!Platform.isTegra(Tree().root))
      for sign in ["/proc/device-tree/compatible", "/etc/nv_tegra_release", "/sys/devices/platform/bus@0/17000000.gpu/load"] {
        let tree = Tree()
        tree.write(sign, sign.hasSuffix("compatible") ? "nvidia,p3768-0000\0nvidia,tegra234\0" : "")
        #expect(Platform.isTegra(tree.root), "\(sign)")
      }
      let pc = Tree()
      pc.write("/sys/firmware/devicetree/base/compatible", "linux,dummy-virt\0")
      #expect(!Platform.isTegra(pc.root))
    }

    @Test("MemAvailable in bytes, 0 when unknown")
    func memory() {
      #expect(Platform.memAvailableBytes(jetson) == 5_362_572 * 1024)
      #expect(Platform.memAvailableBytes(Tree().root) == 0)
    }

    @Test("The cache: $JETLINK_CACHE, else the Jetson's partition, else XDG's")
    func cache() {
      let pc = Tree().root
      #expect(Platform.defaultCache(environment: ["JETLINK_CACHE": "/data/jl"], root: jetson).path == "/data/jl")
      #expect(Platform.defaultCache(environment: [:], root: jetson).path == "/mnt/data/jetlink")
      #expect(Platform.defaultCache(environment: ["XDG_CACHE_HOME": "/xdg"], root: pc).path == "/xdg/jetlink")
      #expect(Platform.defaultCache(environment: ["XDG_CACHE_HOME": ""], root: pc).path.hasSuffix("/.cache/jetlink"))
    }

    @Test("Suspend needs /sys/power/state")
    func suspend() {
      #expect(Platform.canSuspend(jetson))
      #expect(!Platform.canSuspend(Tree().root))
    }

    @Test("Kernel files: trimmed text, integers, and nil for what cannot be read")
    func sysfs() throws {
      #expect(Sysfs.read(jetson.path("/sys/power/mem_sleep")) == "s2idle [deep]")
      #expect(Sysfs.readInt(jetson.path("/sys/class/hwmon/hwmon2/rpm")) == 2367)
      #expect(Sysfs.read(jetson.path("/sys/devices/virtual/thermal/thermal_zone2/temp")) == nil)
      #expect(Sysfs.readInt(jetson.path("/sys/power/mem_sleep")) == nil)
      let tree = Tree()
      tree.write("/attr", "old value\n")
      try Sysfs.write(tree.path("/attr"), "new")
      #expect(tree.read("/attr") == "new")
      let error = #expect(throws: KernelError.self) { try Sysfs.write(tree.path("/missing"), "x") }
      #expect(error?.errno == ENOENT)
    }
  }
#endif
