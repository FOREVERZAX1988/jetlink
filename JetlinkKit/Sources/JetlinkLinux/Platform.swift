#if os(Linux)
  import Foundation
  import Glibc
  import JetlinkServer

  /// Where this host's kernel files are: `/` on a real host, a captured or
  /// made-up tree in the tests. Every path the Linux host reads or writes
  /// goes through one of these.
  public struct HostRoot: Sendable, Equatable {
    /// Put in front of every absolute path: "" for the real host.
    public let prefix: String

    public static let system = HostRoot("")

    public init(_ directory: String) {
      var prefix = directory
      while prefix.hasSuffix("/") { prefix.removeLast() }
      self.prefix = prefix
    }

    public func path(_ absolute: String) -> String {
      prefix + absolute
    }

    /// The names in a directory, sorted as Python's `sorted(glob)` sorts
    /// them; empty when it cannot be listed.
    public func list(_ absolute: String) -> [String] {
      ((try? FileManager.default.contentsOfDirectory(atPath: path(absolute))) ?? []).sorted()
    }

    public func exists(_ absolute: String) -> Bool {
      FileManager.default.fileExists(atPath: path(absolute))
    }
  }

  /// A kernel file or call that failed, and its errno: which says, for one,
  /// whether a suspend is worth trying again (sleep.py's split).
  public struct KernelError: Error, CustomStringConvertible {
    public let what: String
    public let errno: Int32

    public var description: String { "\(what): \(String(cString: strerror(errno)))" }
  }

  /// Small kernel files, read and written with one call each and no
  /// Foundation in between: the sampler reads a handful ten times a second,
  /// and a write's errno decides what the sleeper does next.
  public enum Sysfs {
    /// The file's text with the newline trimmed, or nil on any error: a
    /// Jetson's cv* thermal zones answer ENODATA.
    public static func read(_ path: String) -> String? {
      let fd = open(path, O_RDONLY | O_CLOEXEC)
      guard fd >= 0 else { return nil }
      defer { close(fd) }
      var bytes: [UInt8] = []
      var chunk = [UInt8](repeating: 0, count: 4096)
      while true {
        let n = chunk.withUnsafeMutableBytes { Glibc.read(fd, $0.baseAddress, $0.count) }
        if n < 0 && errno == EINTR { continue }
        guard n >= 0 else { return nil }
        if n == 0 { break }
        bytes += chunk[..<n]
      }
      return String(decoding: bytes, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// A decimal integer file, or nil.
    public static func readInt(_ path: String) -> Int? {
      read(path).flatMap { Int($0) }
    }

    /// Writes `text` to an existing file in one write, as the kernel wants a
    /// sysfs attribute written; throws with the errno.
    public static func write(_ path: String, _ text: String) throws(KernelError) {
      let fd = open(path, O_WRONLY | O_TRUNC | O_CLOEXEC)
      guard fd >= 0 else { throw KernelError(what: path, errno: errno) }
      defer { close(fd) }
      let bytes = Array(text.utf8)
      let n = bytes.withUnsafeBytes { Glibc.write(fd, $0.baseAddress, $0.count) }
      if n < 0 { throw KernelError(what: path, errno: errno) }
    }
  }

  /// The few things the server has to know about the machine it runs on
  /// (Python's `jetlink/server/platform.py`). Each fails open: a wrong guess
  /// costs a cache directory in an odd place, never a refusal to serve.
  public enum Platform {
    /// Where a Jetson keeps the cache: its data partition.
    public static let jetsonCache = "/mnt/data/jetlink"

    /// Any one of these says Tegra: the device tree's compatible string, the
    /// L4T release file, or the Orin's GPU node.
    public static func isTegra(_ root: HostRoot = .system) -> Bool {
      for file in ["/sys/firmware/devicetree/base/compatible", "/proc/device-tree/compatible"] {
        if let data = FileManager.default.contents(atPath: root.path(file)), String(decoding: data, as: UTF8.self).lowercased().contains("tegra") {
          return true
        }
      }
      return ["/etc/nv_tegra_release", "/sys/devices/platform/bus@0/17000000.gpu"].contains { root.exists($0) }
    }

    /// MemAvailable in bytes, or 0 for "no idea". Swap does not count: on a
    /// Tegra the GPU's memory is pinned system RAM and cannot page out.
    public static func memAvailableBytes(_ root: HostRoot = .system) -> Int {
      guard let text = try? String(contentsOfFile: root.path("/proc/meminfo"), encoding: .utf8) else { return 0 }
      for line in text.split(separator: "\n") where line.hasPrefix("MemAvailable:") {
        let fields = line.split(separator: " ")
        if fields.count >= 2, let kb = Int(fields[1]) { return kb * 1024 }
      }
      return 0
    }

    /// $JETLINK_CACHE, else the Jetson's data partition when it exists or
    /// this is a Tegra, else ${XDG_CACHE_HOME:-~/.cache}/jetlink.
    public static func defaultCache(environment: [String: String] = ProcessInfo.processInfo.environment, root: HostRoot = .system) -> URL {
      if let named = environment["JETLINK_CACHE"], !named.isEmpty {
        return URL(fileURLWithPath: named, isDirectory: true)
      }
      var isDirectory: ObjCBool = false
      if FileManager.default.fileExists(atPath: root.path(jetsonCache), isDirectory: &isDirectory) && isDirectory.boolValue || isTegra(root) {
        return URL(fileURLWithPath: jetsonCache, isDirectory: true)
      }
      let base =
        environment["XDG_CACHE_HOME"].flatMap { $0.isEmpty ? nil : URL(fileURLWithPath: $0, isDirectory: true) }
        ?? FileManager.default.homeDirectoryForCurrentUser.appending(path: ".cache", directoryHint: .isDirectory)
      return base.appending(path: "jetlink", directoryHint: .isDirectory)
    }

    /// Whether --sleep-after has a kernel to ask.
    public static func canSuspend(_ root: HostRoot = .system) -> Bool {
      root.exists("/sys/power/state")
    }

    /// `name` found on `path` (a PATH value) as an executable, or nil.
    static func which(_ name: String, path: String?) -> String? {
      for directory in (path ?? "/usr/sbin:/usr/bin:/sbin:/bin").split(separator: ":") where !directory.isEmpty {
        let candidate = "\(directory)/\(name)"
        if access(candidate, X_OK) == 0 { return candidate }
      }
      return nil
    }
  }

  /// Where the Linux host's own lines go: the server's log, or a test's list.
  typealias LinuxLog = @Sendable (Log.Level, String) -> Void

  func serverLog(_ category: String) -> LinuxLog {
    let log = ServerLog(category: category)
    return { level, message in
      switch level {
      case .info: log.info(message)
      case .warning: log.warning(message)
      case .error: log.error(message)
      }
    }
  }
#endif
