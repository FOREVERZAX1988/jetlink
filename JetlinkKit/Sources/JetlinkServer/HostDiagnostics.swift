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

    /// This process's QNN and onnxruntime libraries, from its own maps. Its
    /// pid is written out: this runs through a shell, and a shell's own
    /// `/proc/self/maps` is the shell's (or the grep's, in the pipe), so a
    /// probe that said `self` answered about itself and found nothing, always.
    private static let mapsCommand =
      "grep -iE 'qnn|onnxruntime|cdsprpc|adsprpc|libjetlink' /proc/\(ProcessInfo.processInfo.processIdentifier)/maps"
      + " | grep -oE '/[^ ]+\\.so' | sort -u"
    /// Where the app's own native libraries are when the maps naming failed,
    /// which is where the QNN libraries it brought are opened from.
    private static let appDirCommand = "ls -d /data/app/*/*jetlink*/lib/arm64 2>/dev/null"
    /// The APK itself, for a build that keeps its libraries inside it, which
    /// AGP's default `extractNativeLibs` off is: the directory above does not
    /// exist then, and every library of the app is mapped as the APK, so the
    /// maps have no `.so` of the app's to name, only `base.apk`.
    private static let apkCommand =
      "grep -oE '/data/app/[^ ]*base\\\\.apk' /proc/\\(ProcessInfo.processInfo.processIdentifier)/maps 2>/dev/null | head -1"
    private static let deviceCommand = "ls -l /dev/adsprpc-smd /dev/cdsprpc-smd /dev/fastrpc-adsp /dev/fastrpc-cdsp /dev/fastrpc-cdsp-secure"
    private static let qnnWords = ["qnn", "onnxruntime", "cdsprpc", "adsprpc", "htp", "skel", "snpe", "hexagon"]
    private static let dspWords = ["skel", "qnn", "dsp", "rfsa", "fastrpc"]
    private static let logWords = ["qnn", "htp", "onnxruntime", "dsp", "adsp", "fastrpc", "fallback", "unsupported", "skel", "stub", "arch", "fail"]
    private static let propertyWords = [
      "soc", "board", "hardware", "platform", "qnn", "hexagon", "dsp", "adsp", "cpu.abi",
      "product.model", "product.device", "build.version", "chipset", "soc_id", "vendor.qti"
    ]

    /// The facts, the ones that could name a reason first.
    private static func gather() -> [String] {
      var lines: [String] = []
      let mapped = shell(mapsCommand)
      lines.append(contentsOf: section("the QNN and onnxruntime libraries this process has open", mapped))
      let carried = appLibraries(mapped)
      if carried.isEmpty {
        lines.append("nothing named the app's own libraries: this process's maps, its APK and an app library directory all came up empty")
      } else {
        // Everything, not only the QNN-named: a Stub the phone cannot use and
        // its absence are the same line here, and the Skel sits beside it.
        lines.append(contentsOf: section("every library the app itself carries", carried))
      }
      let arches = htpArches(carried)
      lines.append("the HTP architectures this build carries: \(arches.isEmpty ? "none named" : arches.joined(separator: " "))")
      let system =
        listing("/vendor/lib64") + listing("/vendor/lib") + listing("/system/lib64")
        + listing("/odm/lib64") + listing("/vendor/lib64/qnn") + listing("/system/lib64/qnn")
      lines.append(contentsOf: section("the phone's own QNN and onnxruntime libraries", matching(system, qnnWords)))
      let dsp =
        listing("/vendor/lib/rfsa/adsp") + listing("/vendor/lib64/rfsa/adsp") + listing("/vendor/lib64/rfsa/cdsp")
        + listing("/vendor/lib/rfsa/cdsp") + listing("/vendor/dsp") + listing("/vendor/dsp/cdsp")
        + listing("/system/lib/rfsa/adsp") + listing("/odm/lib/rfsa/adsp")
      lines.append(contentsOf: section("the phone's DSP libraries, the Skel a Hexagon loads", matching(dsp, dspWords)))
      lines.append(contentsOf: section("the DSP devices this process talks on", shell(deviceCommand)))
      lines.append(contentsOf: section("the chip", matching(shell("getprop"), propertyWords)))
      lines.append(contentsOf: section("this process's own log, its pid's lines", processLog()))
      return lines
    }

    /// What the app's own QNN comes in. The libraries beside `libjetlink.so`
    /// where the loader extracted them; else what the APK carries, which
    /// `unzip` reads out of it; else an app library directory, for a build
    /// extracted somewhere else. Empty when nothing could name them.
    private static func appLibraries(_ mapped: [String]) -> [String] {
      if let directory = mapped.first(where: { $0.hasSuffix("/libjetlink.so") })
        .map({ ($0 as NSString).deletingLastPathComponent })
      {
        return listing(directory)
      }
      if let apk = shell(apkCommand).first, apk.hasSuffix(".apk") {
        let carried = shell("unzip -l '\(apk)' 2>/dev/null | grep -oE 'lib/arm64-v8a/[^ ]+' | sort -u")
        if !carried.isEmpty { return carried }
      }
      if let directory = shell(appDirCommand).first { return listing(directory) }
      return []
    }

    /// The HTP architectures the app's own QNN carries, which is every arch an
    /// NPU build can be compiled for here: a Stub and its Skel are named for
    /// one. A Hexagon newer than the last of these is one this build cannot
    /// run on at all, which is the whole answer when an NPU build takes no
    /// node of its graph. "V68 V69 V73 V75 V79 V81" from the
    /// `libQnnHtpV68Stub.so` upwards that the app carries.
    private static func htpArches(_ libraries: [String]) -> [String] {
      var arches: Set<Int> = []
      for library in libraries {
        guard let start = library.range(of: "libQnnHtpV")?.upperBound else { continue }
        let digits = library[start...].prefix { $0.isNumber }
        let rest = library[start...].dropFirst(digits.count)
        guard !digits.isEmpty, rest.hasPrefix("Stub.so"), let arch = Int(digits) else { continue }
        arches.insert(arch)
      }
      return arches.sorted().map { "V\($0)" }
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
