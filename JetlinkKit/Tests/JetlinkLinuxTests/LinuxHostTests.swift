#if os(Linux)
  import Foundation
  import JetlinkLinux
  import JetlinkServer
  import Testing

  @Suite("Linux host")
  struct LinuxHostTests {
    @Test("Until this host can sleep, the hello says it never does")
    func neverSleepsYet() {
      #expect(LinuxHost.hooks(cache: URL(fileURLWithPath: "/tmp"), sleepAfter: 900).sleepAfter == 0)
    }
  }
#endif
