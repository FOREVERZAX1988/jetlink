#if os(Linux)
  import Foundation
  import Glibc
  import Testing

  @testable import JetlinkLinux

  @Suite("Power off")
  struct PowerOffTests {
    func run(tegra: Bool, dryRun: Bool) -> (calls: Int, lines: Lines) {
      let tree = Tree()
      if dryRun { tree.write("/\(PowerOff.dryRunName)", "") }
      let calls = Dial(0)
      let lines = Lines()
      let hook = PowerOff.hook(cache: tree.url, tegra: tegra, powerOff: { _ in calls.value += 1 }, log: lines.log)
      // Always accepted: the reply says ok and "powering off" either way.
      let action = hook("car battery")
      #expect(action != nil)
      #expect(calls.value == 0)
      action?()
      return (calls.value, lines)
    }

    @Test("A Jetson powers off once the reply is out")
    func jetson() {
      let (calls, lines) = run(tegra: true, dryRun: false)
      #expect(calls == 1)
      #expect(lines.has(.warning, "powering off"))
    }

    @Test("The dry-run file keeps a Jetson up")
    func dryRun() {
      let (calls, lines) = run(tegra: true, dryRun: true)
      #expect(calls == 0)
      #expect(lines.has(.warning, "dry run: staying up"))
    }

    @Test("A PC never powers off", arguments: [false, true])
    func pc(dryRun: Bool) {
      let (calls, lines) = run(tegra: false, dryRun: dryRun)
      #expect(calls == 0)
      #expect(lines.has(.warning, "staying up"))
    }

    @Test("systemctl is looked up on PATH")
    func which() {
      let tree = Tree()
      tree.write("/a/systemctl", "not executable")
      tree.write("/b/systemctl", "#!/bin/sh\n")
      chmod(tree.path("/b/systemctl"), 0o755)
      #expect(Platform.which("systemctl", path: "\(tree.path("/a")):\(tree.path("/b"))") == tree.path("/b/systemctl"))
      #expect(Platform.which("systemctl", path: tree.path("/a")) == nil)
    }

    @Test("The child starts with no signal blocked and the stop signals at their defaults")
    func childSignals() throws {
      let tree = Tree()
      let out = tree.path("/status")
      // As the daemon's threads are: SIGINT and SIGTERM blocked, SIGPIPE ignored.
      var held = sigset_t()
      sigemptyset(&held)
      sigaddset(&held, SIGINT)
      sigaddset(&held, SIGTERM)
      var previous = sigset_t()
      pthread_sigmask(SIG_BLOCK, &held, &previous)
      defer { pthread_sigmask(SIG_SETMASK, &previous, nil) }
      // Left ignored: other suites write to sockets in this process.
      signal(SIGPIPE, SIG_IGN)

      let pid = try PowerOff.spawn("/bin/sh", ["-c", "grep -E '^Sig(Blk|Ign)' /proc/self/status > \(out)"])
      #expect(PowerOff.exitStatus(of: pid) == 0)
      let status = try String(contentsOfFile: out, encoding: .utf8)
      let masks = Dictionary(
        uniqueKeysWithValues: status.split(separator: "\n").map { line in
          let parts = line.split(separator: ":")
          return (String(parts[0]), UInt64(parts[1].trimmingCharacters(in: .whitespaces), radix: 16)!)
        })
      #expect(masks["SigBlk"] == 0)
      for number in [SIGINT, SIGTERM, SIGPIPE] {
        #expect(masks["SigIgn"]! & (1 << UInt64(number - 1)) == 0)
      }
    }

    @Test("A failing child's status comes back")
    func exitStatus() throws {
      #expect(PowerOff.exitStatus(of: try PowerOff.spawn("/bin/sh", ["-c", "exit 3"])) == 3)
      #expect(throws: KernelError.self) { try PowerOff.spawn("/nonexistent/systemctl", ["poweroff"]) }
    }
  }
#endif
