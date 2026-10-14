import Foundation

#if canImport(Android)
  import Android
#endif

/// What a phone's neural runtime looks like, in lines for a log someone can
/// paste.
///
/// QNN refuses a model in silence: a Stub for another Hexagon, a Skel the DSP
/// will not load, a device the app may not open, and onnxruntime runs the
/// whole graph on the CPU without an error. The reason is never in the model,
/// so this writes what the phone says instead: the QNN libraries the app
/// brought and the phone has, the chip, and this process's own log.
///
/// Android only. Everywhere else it is a no-op, so a caller needs no guard of
/// its own.
package enum HostDiagnostics {
  /// The most lines one report writes, so a pasted log stays readable.
  package static let lineLimit = 200

  /// Writes `reason`, then what this machine says about its NPU runtime.
  ///
  /// For the engine's thread, where a second costs nothing, and never for a
  /// frame's. It shells out, so it takes about that second. Two reports inside
  /// a minute are one: a build that wrote no context is followed at once by
  /// the load that pays for it, which would repeat everything.
  package static func report(_ reason: String, limit: Int = lineLimit) {
    #if os(Android)
      let log = ServerLog(category: "npu")
      guard claim() else {
        log.info("the phone's NPU runtime was read a moment ago, not again for: \(reason)")
        return
      }
      log.warning("the phone's NPU runtime, as its libraries and its log tell it: \(reason)")
      let lines = gather()
      for line in lines.prefix(limit) {
        log.info(line)
      }
      if lines.count > limit {
        log.info("... and \(lines.count - limit) more lines of it")
      }
    #else
      _ = reason
      _ = limit
    #endif
  }

  #if os(Android)
    private static let lock = NSLock()
    private static let spacing: TimeInterval = 60
    nonisolated(unsafe) private static var lastProbe = Date.distantPast

    /// True for the first report in `spacing`, which holds the rest off.
    private static func claim() -> Bool {
      lock.lock()
      defer { lock.unlock() }
      let now = Date()
      guard now.timeIntervalSince(lastProbe) >= spacing else { return false }
      lastProbe = now
      return true
    }

    /// This process's QNN and onnxruntime libraries, from its own maps.
    private static let mapsCommand = "grep -iE 'qnn|onnxruntime|cdsprpc|adsprpc|libjetlink' /proc/self/maps | grep -oE '/[^ ]+\\.so' | sort -u"
    private static let deviceCommand = "ls -l /dev/adsprpc-smd /dev/cdsprpc-smd /dev/fastrpc-adsp /dev/fastrpc-cdsp"
    private static let qnnWords = ["qnn", "onnxruntime", "cdsprpc", "adsprpc"]
    private static let dspWords = ["skel", "qnn", "dsp", "rfsa", "fastrpc"]
    private static let logWords = ["qnn", "htp", "onnxruntime", "dsp", "adsp", "fastrpc", "fallback", "unsupported"]
    private static let propertyWords = [
      "soc", "board", "hardware", "platform", "qnn", "hexagon", "dsp", "adsp", "cpu.abi",
      "product.model", "product.device", "build.version"
    ]

    /// The facts, the ones that could name a reason first.
    private static func gather() -> [String] {
      var lines: [String] = []
      let mapped = shell(mapsCommand)
      lines.append(contentsOf: section("the QNN and onnxruntime libraries this process has open", mapped))
      let directory = mapped.first { $0.hasSuffix("/libjetlink.so") }.map { ($0 as NSString).deletingLastPathComponent }
      if let directory {
        lines.append(contentsOf: section("what the app brought in \(directory)", matching(listing(directory), qnnWords)))
      } else {
        lines.append("no libjetlink.so in this process's maps, so no library directory to name")
      }
      let system = listing("/vendor/lib64") + listing("/vendor/lib") + listing("/system/lib64")
      lines.append(contentsOf: section("the phone's own QNN and onnxruntime libraries", matching(system, qnnWords)))
      let dsp = listing("/vendor/lib/rfsa/adsp") + listing("/vendor/lib64/rfsa/adsp") + listing("/vendor/dsp") + listing("/system/lib/rfsa/adsp")
      lines.append(contentsOf: section("the phone's DSP libraries, the Skel a Hexagon loads", matching(dsp, dspWords)))
      lines.append(contentsOf: section("the DSP devices this process talks on", shell(deviceCommand)))
      lines.append(contentsOf: section("the chip", matching(shell("getprop"), propertyWords)))
      lines.append(contentsOf: section("this process's own log, its pid's lines", processLog()))
      return lines
    }

    /// A title and the lines under it, or the one line that says it is empty.
    private static func section(_ title: String, _ body: [String]) -> [String] {
      guard !body.isEmpty else { return ["\(title): nothing"] }
      return ["\(title):"] + body.map { "  \($0)" }
    }

    /// The lines of `lines` that say one of `words`, in the order they came.
    private static func matching(_ lines: [String], _ words: [String]) -> [String] {
      lines.filter { line in
        let lower = line.lowercased()
        return words.contains { lower.contains($0) }
      }
    }

    /// One `ls -l` of a directory, as lines, or its own line saying why not.
    private static func listing(_ path: String) -> [String] {
      shell("ls -l '\(path)'").filter { !$0.hasPrefix("total ") }
    }

    /// This process's lines in logcat, kept to what a neural runtime says: an
    /// app may read its own alone, and jetlink's own tag is left out.
    private static func processLog() -> [String] {
      let all = shell("logcat -d -v epoch -t 3000")
      let pid = String(ProcessInfo.processInfo.processIdentifier)
      var told: [String] = []
      for line in all {
        // "1791043556.899  3544  3782 W litert  : message"
        let fields = line.split(separator: " ", maxSplits: 4, omittingEmptySubsequences: true)
        guard fields.count == 5, String(fields[1]) == pid, let colon = fields[4].range(of: ": ") else { continue }
        let tag = fields[4][..<colon.lowerBound].trimmingCharacters(in: .whitespaces)
        guard tag != "jetlink" else { continue }
        let text = "\(tag): \(fields[4][colon.upperBound...])"
        let lower = text.lowercased()
        guard logWords.contains(where: { lower.contains($0) }) else { continue }
        told.append(text)
      }
      if told.isEmpty { return Array(all.suffix(20)) }
      return Array(told.suffix(80))
    }

    /// A shell command's output, both streams, as lines, capped.
    private static func shell(_ command: String, cap: Int = 1 << 18) -> [String] {
      guard let pipe = popen("\(command) 2>&1", "r") else { return ["cannot run \(command)"] }
      defer { pclose(pipe) }
      var data = Data()
      var buffer = [UInt8](repeating: 0, count: 1 << 16)
      while data.count < cap {
        let n = fread(&buffer, 1, buffer.count, pipe)
        if n <= 0 { break }
        data.append(contentsOf: buffer[..<n])
      }
      let text = String(decoding: data, as: UTF8.self)
      return text.split(separator: "\n").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
    }
  #endif
}
