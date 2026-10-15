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

    /// The app's own directory of libraries, as the Android app names it
    /// (`ApplicationInfo.nativeLibraryDir`), handed over at start. A probe
    /// cannot find it on its own: `/data/app` may be walked but not read by an
    /// app (mode 0711), so a glob over it matches nothing, and Android carries
    /// no `unzip` to read the APK's entries instead. Both failures read as
    /// "this build carries no Stub", which is why the directory is passed in.
    nonisolated(unsafe) package static var nativeDirectory: String?

    /// True for the first report in `spacing`, which holds the rest off.
    private static func claim() -> Bool {
      lock.lock()
      defer { lock.unlock() }
      let now = Date()
      guard now.timeIntervalSince(lastProbe) >= spacing else { return false }
      lastProbe = now
      return true
    }

    /// These probes run through `popen`, one shell each, and every one of them
    /// names this process's pid: a shell's own `/proc/self/maps` is the shell's
    /// (or the grep's, in the pipe), so a probe that said `self` answered about
    /// itself and found nothing, always.
    private static var pid: String { String(ProcessInfo.processInfo.processIdentifier) }
    /// Two lines of the process this report is about, and how many mappings it
    /// has: a probe that read somewhere else says so here.
    private static var processCommand: String { "head -2 /proc/\(pid)/status; wc -l /proc/\(pid)/maps" }
    /// Every QNN, DSP and onnxruntime library this process has open, and where
    /// each came from: the app's own directory, or the phone's.
    private static var mapsCommand: String {
      "grep -iE 'qnn|onnxruntime|cdsprpc|adsprpc|libjetlink' /proc/\(pid)/maps | grep -oE '/[^ ]+\\.so' | sort -u"
    }
    /// The APK this process runs from: `pm` answers for any package, so the
    /// app's own directory is one `ls` away even when no config arrived.
    private static var packageCommand: String {
      "p=$(cat /proc/\(pid)/cmdline | tr -c '[:print:]' '\\n' | head -1); pm path \"$p\""
    }
    private static let deviceCommand = "ls -l /dev/adsprpc-smd /dev/cdsprpc-smd /dev/fastrpc-adsp /dev/fastrpc-cdsp /dev/fastrpc-cdsp-secure"
    /// The libraries a QNN build has to agree on with itself: the backend, the
    /// system library and the pair for this phone's Hexagon.
    private static let qnnFiles = ["libQnnHtp.so", "libQnnSystem.so", "libQnnHtpV85Stub.so", "libQnnHtpV85Skel.so"]
    private static let qnnWords = ["qnn", "onnxruntime", "cdsprpc", "adsprpc", "htp", "skel", "snpe", "hexagon"]
    private static let dspWords = ["skel", "qnn", "dsp", "rfsa", "fastrpc"]
    private static let logWords = ["qnn", "htp", "onnxruntime", "dsp", "adsp", "fastrpc", "fallback", "unsupported", "skel", "stub", "arch", "fail"]
    /// What a neural runtime calls itself, for the lines that are its own
    /// whatever process wrote them.
    private static let runtimeWords = ["onnxruntime", "qnn", "adsprpc", "cdsprpc", "fastrpc", "hexagon", "htp", "skel", "stub"]
    private static let propertyWords = [
      "soc", "board", "hardware", "platform", "qnn", "hexagon", "dsp", "adsp", "cpu.abi",
      "product.model", "product.device", "build.version", "chipset", "soc_id", "vendor.qti",
    ]

    /// The facts, the ones that could name a reason first.
    private static func gather() -> [String] {
      var lines: [String] = []
      lines.append(contentsOf: section("this process", shell(processCommand)))
      let mapped = shell(mapsCommand)
      lines.append(contentsOf: section("the QNN and onnxruntime libraries this process has open", mapped))
      let carried = appLibraries(mapped)
      if carried.isEmpty {
        lines.append("nothing named the app's own libraries: this process's maps, the app's own directory and `pm path` all came up empty")
      } else {
        // Everything, not only the QNN-named: a Stub the phone cannot use and
        // its absence are the same line here, and the Skel sits beside it.
        lines.append(contentsOf: section("every library the app itself carries", carried))
      }
      let arches = htpArches(carried)
      lines.append("the HTP architectures this build carries: \(arches.isEmpty ? "none named" : arches.joined(separator: " "))")
      if !carried.contains(where: { $0.contains("Skel.so") }) {
        lines.append("no Skel beside the Stub: the DSP has none to load, and QNN takes no node of any model")
      }
      lines.append(contentsOf: section("the app's own QNN, hashed", hashes(mapped)))
      lines.append(contentsOf: section("where the Skel is looked for, ADSP_LIBRARY_PATH", skelSearch()))
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
      lines.append(contentsOf: section("the neural runtime's own lines, whatever pid wrote them", runtimeLog()))
      return lines
    }

    /// The directory the app's own libraries are in: the one the process mapped
    /// `libjetlink.so` from, else the one the app handed over at start, else
    /// the installed APK's `lib/<abi>` beside it. Nil when nothing named it.
    private static func appDirectory(_ mapped: [String]) -> String? {
      if let library = mapped.first(where: { $0.hasSuffix("/libjetlink.so") }) {
        return (library as NSString).deletingLastPathComponent
      }
      if let directory = nativeDirectory, !directory.isEmpty {
        return directory
      }
      let found = shell(packageCommand).first { $0.hasPrefix("package:") }
      guard let apk = found.map({ String($0.dropFirst("package:".count)) }), apk.hasSuffix(".apk") else {
        return nil
      }
      let base = (apk as NSString).deletingLastPathComponent
      return shell("ls -d '\(base)/lib/'*").first { $0.hasPrefix("/") } ?? base
    }

    /// What the app's own QNN comes in, as one `ls` of that directory.
    private static func appLibraries(_ mapped: [String]) -> [String] {
      guard let directory = appDirectory(mapped) else { return [] }
      return listing(directory)
    }

    /// The same files, hashed: a phone whose copies are not the APK's is
    /// running another QNN build, which no setting and no preparation fixes.
    private static func hashes(_ mapped: [String]) -> [String] {
      guard let directory = appDirectory(mapped) else { return [] }
      let files = qnnFiles.map { "'\(directory)/\($0)'" }.joined(separator: " ")
      return shell("sha256sum \(files) | sed 's#.*/##'")
    }

    /// `ADSP_LIBRARY_PATH`, which the app sets to its own directory and the
    /// system's DSP ones: the Skel is loaded from one of them, and a path that
    /// does not exist is one the Hexagon cannot be handed a Skel from.
    private static func skelSearch() -> [String] {
      let value = shell("printenv ADSP_LIBRARY_PATH")
      var lines = value
      for directory in (value.first ?? "").split(separator: ";").map(String.init) where !directory.isEmpty {
        let found = shell("ls -l '\(directory)' 2>&1 | grep -iE 'skel|stub' | sed 's/.* //'")
        lines.append("\(directory): \(found.isEmpty ? "no Skel, no Stub" : found.joined(separator: " / "))")
      }
      return lines
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
      var told: [String] = []
      for line in all {
        guard let line = fields(line), line.pid == pid, line.tag != "jetlink" else { continue }
        let text = "\(line.tag): \(line.text)"
        guard logWords.contains(where: { text.lowercased().contains($0) }) else { continue }
        told.append(text)
      }
      if told.isEmpty { return Array(all.suffix(20)) }
      return Array(told.suffix(80))
    }

    /// What the phone's neural runtime said itself, whatever process said it:
    /// QNN's own line when a provider takes no node, the DSP's, and
    /// onnxruntime's. jetlink's own lines are above, and are left out here.
    private static func runtimeLog() -> [String] {
      let all = shell("logcat -d -v epoch -t 4000")
      let told = all.filter { line in
        guard let line = fields(line), line.tag != "jetlink" else { return false }
        let text = "\(line.tag): \(line.text)"
        return runtimeWords.contains { text.lowercased().contains($0) }
      }
      return Array(told.suffix(40))
    }

    /// "1791043556.899  3544  3782 W litert  : message" → its pid, tag and text.
    private static func fields(_ line: String) -> (pid: String, tag: String, text: String)? {
      let parts = line.split(separator: " ", maxSplits: 4, omittingEmptySubsequences: true)
      guard parts.count == 5, let colon = parts[4].range(of: ": ") else { return nil }
      let tag = parts[4][..<colon.lowerBound].trimmingCharacters(in: .whitespaces)
      return (String(parts[1]), tag, String(parts[4][colon.upperBound...]))
    }

    /// A shell command's output, both streams, as lines, capped. The whole
    /// command's stderr is redirected, not the last one's alone: in a pipeline
    /// `a | b 2>&1` leaves `a`'s failure in the shell's own stderr, which
    /// nothing reads, so a probe that could not read its file answered with
    /// nothing, which is what a report must never say.
    private static func shell(_ command: String, cap: Int = 1 << 18) -> [String] {
      guard let pipe = popen("( \(command) ) 2>&1", "r") else { return ["cannot run \(command)"] }
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
