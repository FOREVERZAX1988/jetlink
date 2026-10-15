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
  /// The most lines one report writes, so a pasted log stays readable. Every
  /// section is capped on its own as well, and the lines a reason lives in come
  /// first: a cap that fell on the whole report once hid the one line that names
  /// the reason, this budget is what the sections at their caps add up to.
  package static let lineLimit = 320

  /// The runtime's own lines, handed over by the backend that brings
  /// onnxruntime (`OrtRuntimeLog.install()`), oldest first. It is the one place
  /// a provider writes its reason for taking no node of a graph -- an Android
  /// app's stderr has no reader -- and nothing else here can supply it. Nil on
  /// a build whose backend carries no such ring, and outside Android's report
  /// it is only carried.
  nonisolated(unsafe) package static var runtimeLines: (() -> [String])?

  /// The QNN backend library this run loads, when the app asks for the phone's
  /// own (`OrtBackend.qnnBackend`): nil is the copy beside the app. The report
  /// prints which of the two, because a backend, a Stub and a Skel from three
  /// QNN builds are three builds that take no node of any model.
  nonisolated(unsafe) package static var qnnBackend: String?

  /// *Phone's Own QNN*, this run. The report prints it: which pair is in force
  /// is what a silent refusal is read against, and it is the one thing a log
  /// pasted from a phone cannot say by itself.
  nonisolated(unsafe) package static var phoneOwnQnn = false

  /// The phone's own libraries this run opened by their paths, before the
  /// backend was handed to onnxruntime. A name opened after them resolves to
  /// them, which is what keeps the phone's own backend from running beside the
  /// app's Stub of the same name (`Host.phoneLibraries`).
  nonisolated(unsafe) package static var phoneLibraries: [String] = []

  /// Writes `reason`, then what this machine says about its NPU runtime.
  ///
  /// For the engine's thread, where a second costs nothing, and never for a
  /// frame's. It shells out, so it takes about that second. Every report is
  /// written, however soon it follows another: a build that wrote no context is
  /// followed at once by the load that pays for it, so one failed start writes
  /// two reports -- and the second is the one a log is pasted from, a person
  /// having retried the profile. A report held back is a reason no one can
  /// read. What is held back instead is the reading of the directories, which
  /// do not change inside a minute (`listing`).
  package static func report(_ reason: String, limit: Int = lineLimit) {
    #if os(Android)
      let log = ServerLog(category: "npu")
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
    /// One `ls -l` per directory per `spacing`, with the time it was read.
    nonisolated(unsafe) private static var listings: [String: (at: Date, lines: [String])] = [:]

    /// The app's own directory of libraries, as the Android app names it
    /// (`ApplicationInfo.nativeLibraryDir`), handed over at start. A probe
    /// cannot find it on its own: `/data/app` may be walked but not read by an
    /// app (mode 0711), so a glob over it matches nothing, and Android carries
    /// no `unzip` to read the APK's entries instead. Both failures read as
    /// "this build carries no Stub", which is why the directory is passed in.
    nonisolated(unsafe) package static var nativeDirectory: String?

    /// These probes run through `popen`, one shell each, and every one of them
    /// names this process's pid: a shell's own `/proc/self/maps` is the shell's
    /// (or the grep's, in the pipe), so a probe that said `self` answered about
    /// itself and found nothing, always. The pid is the C library's, which is
    /// this process by definition, and `processFacts` prints it beside the one
    /// Foundation names and the one the shell's own parent has: a report whose
    /// pid is not a process read nothing anywhere below, so the three are
    /// evidence too, and the empty sections are then no evidence at all.
    private static var pid: String { String(getpid()) }
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
    private static let logWords = [
      "qnn", "htp", "onnxruntime", "dsp", "adsp", "fastrpc", "fallback", "unsupported", "skel", "stub", "arch", "fail",
      "device", "convert", "version", "soc",
    ]
    /// What a neural runtime calls itself, for the lines that are its own
    /// whatever process wrote them.
    private static let runtimeWords = [
      "onnxruntime", "qnn", "adsprpc", "cdsprpc", "fastrpc", "hexagon", "htp", "skel", "stub", "device", "version", "arch",
    ]
    private static let propertyWords = [
      "soc", "board", "hardware", "platform", "qnn", "hexagon", "dsp", "adsp", "cpu.abi",
      "product.model", "product.device", "build.version", "chipset", "soc_id", "vendor.qti",
    ]

    /// The facts, the ones that could name a reason first.
    private static func gather() -> [String] {
      var lines: [String] = []
      // The reason first: a provider that takes no node of a graph writes why
      // under onnxruntime's logger, and the phone's log holds what QNN, the DSP
      // and the process itself said around it. The directories come after,
      // each within its own budget, so nothing above them can be cut off.
      // Every line of it, not the first of a cap: QNN writes its refusal
      // *after* the graph transformers -- the device is created at the
      // partitioning that follows them -- so a cap of 24 lines showed four
      // transformers and cut the one line this report exists for. The ring is
      // what onnxruntime wrote since the session was made; it is short, and it
      // is the section that answers the question.
      lines.append(
        contentsOf: section(
          "the runtime's own lines, as onnxruntime and QNN wrote them", runtimeLines?() ?? [], cap: 200))
      lines.append(contentsOf: section("this process's own log, its pid's lines", processLog(), cap: 80))
      lines.append(contentsOf: section("the runtime's lines in the phone's log, whatever pid wrote them", runtimeLog(), cap: 40))
      lines.append(contentsOf: section("this process", processFacts()))
      let mapped = mappedLibraries()
      lines.append(contentsOf: section("the QNN and onnxruntime libraries this process has open", mapped))
      let carried = appLibraries(mapped)
      if carried.isEmpty {
        lines.append("nothing named the app's own libraries: this process's maps, the app's own directory and `pm path` all came up empty")
      } else {
        // Everything, not only the QNN-named: a Stub the phone cannot use and
        // its absence are the same line here, and the Skel sits beside it.
        lines.append(contentsOf: section("every library the app itself carries", carried, cap: 24))
      }
      let arches = htpArches(carried)
      lines.append("the HTP architectures this build carries: \(arches.isEmpty ? "none named" : arches.joined(separator: " "))")
      if !carried.contains(where: { $0.contains("Skel.so") }) {
        lines.append("no Skel beside the Stub: the DSP has none to load, and QNN takes no node of any model")
      }
      let search: [(directory: String, files: [String])] = adspDirectories().map { directory in
        (directory: directory, files: named(directory))
      }
      lines.append(contentsOf: section("where the Skel is looked for, ADSP_LIBRARY_PATH, in order", skelSearch(search)))
      lines.append(contentsOf: section("the pair this run is set to use, and the phone's own", pairInForce(search, arch: arches.last)))
      lines.append(contentsOf: section("the app's own QNN, hashed", hashes(mapped)))
      let system =
        listing("/vendor/lib64") + listing("/vendor/lib") + listing("/system/lib64")
        + listing("/odm/lib64") + listing("/vendor/lib64/qnn") + listing("/system/lib64/qnn")
      lines.append(contentsOf: section("the phone's own QNN and onnxruntime libraries", matching(system, qnnWords), cap: 14))
      let dsp =
        listing("/vendor/lib/rfsa/adsp") + listing("/vendor/lib64/rfsa/adsp") + listing("/vendor/lib64/rfsa/cdsp")
        + listing("/vendor/lib/rfsa/cdsp") + listing("/vendor/dsp") + listing("/vendor/dsp/cdsp")
        + listing("/system/lib/rfsa/adsp") + listing("/odm/lib/rfsa/adsp")
      lines.append(contentsOf: section("the phone's DSP libraries, the Skel a Hexagon loads", matching(dsp, dspWords), cap: 14))
      lines.append(contentsOf: section("the DSP devices this process talks on", shell(deviceCommand)))
      lines.append(contentsOf: section("the chip, its properties", matching(shell("getprop"), propertyWords), cap: 14))
      return lines
    }

    /// This process, as three readings of one pid: the C library's, which is
    /// this process by definition; Foundation's, printed for comparison; and
    /// the one the shell that runs every probe has for a parent. A report whose
    /// pid is not a process has read nothing anywhere, and says so here rather
    /// than three sections below in silence.
    private static func processFacts() -> [String] {
      var lines = [
        "getpid(): \(getpid())",
        "ProcessInfo.processIdentifier: \(ProcessInfo.processInfo.processIdentifier)",
        // The shell that runs these probes is a child of this process, so the
        // parent of the shell's own parent is this pid: a reading that goes
        // through neither Foundation nor the C library.
        "this process, as the shell reads it: \(shell("awk '{print $4}' /proc/$PPID/stat").first ?? "unread")",
        // The domain SELinux puts this process in: the subject of every
        // `avc: denied` line in the log, and the whole reason one phone
        // refuses its DSP to the app and another hands it over.
        "this app's SELinux domain, as the kernel reads it: \(selinuxDomain())",
      ]
      lines.append(contentsOf: shell(processCommand))
      return lines
    }

    /// The SELinux domain this process runs in, from the kernel's own file for
    /// it: `u:r:untrusted_app_34:s0:c199,c257,c512,c768` on the app an OEM's
    /// firmware refuses its DSP to, `u:r:platform_app:s0` on one it does not.
    /// "unread" where the phone has no SELinux to put a process in one.
    private static func selinuxDomain() -> String {
      (try? String(contentsOfFile: "/proc/self/attr/current", encoding: .utf8))
        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) } ?? "unread"
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

    /// Every QNN, DSP and onnxruntime library this process has open, read
    /// here, in this process, before any shell is asked: the probe below names
    /// a pid, and a pid and `/proc/self` are not the same question -- on the
    /// phone this was written for the pid's maps answered "No such file or
    /// directory" and the section came back empty, which reads as "the app
    /// carried no QNN at all". `/proc/self` is the process the report is
    /// written in, whatever a pid says.
    private static func mappedLibraries() -> [String] {
      let here = selfMaps()
      return here.isEmpty ? shell(mapsCommand) : here
    }

    private static func selfMaps() -> [String] {
      guard let maps = FileManager.default.contents(atPath: "/proc/self/maps"),
        let text = String(data: maps, encoding: .utf8)
      else { return [] }
      var paths: Set<String> = []
      for line in text.split(separator: "\n") {
        guard let field = line.split(separator: " ").first(where: { $0.hasPrefix("/") }) else { continue }
        let lower = field.lowercased()
        guard qnnWords.contains(where: { lower.contains($0) }) else { continue }
        paths.insert(String(field))
      }
      return paths.sorted()
    }

    /// The same files, hashed: a phone whose copies are not the APK's is
    /// running another QNN build, which no setting and no preparation fixes.
    /// Sixteen hex characters and the name, not the whole of either: two builds
    /// of one library are the same size (the AAR's 14,058,420-byte Skel and the
    /// SDK's differ in 503 bytes, and in nothing else), so the hash is the only
    /// line that tells them apart, and the directory is the same for all four.
    private static func hashes(_ mapped: [String]) -> [String] {
      guard let directory = appDirectory(mapped) else { return [] }
      let files = qnnFiles.map { "'\(directory)/\($0)'" }.joined(separator: " ")
      return shell("sha256sum \(files) | awk '{n=split($2,a,\"/\"); print substr($1,1,16), a[n]}'")
    }

    /// `ADSP_LIBRARY_PATH`'s directories, in the order the DSP searches them:
    /// the Skel is loaded from the first that holds one, so the order decides
    /// which of two QNN builds a phone's Hexagon is handed. The app's own
    /// directory comes first, the system's DSP ones after it.
    private static func adspDirectories() -> [String] {
      (shell("printenv ADSP_LIBRARY_PATH").first ?? "").split(separator: ";").map(String.init).filter { !$0.isEmpty }
    }

    /// The Skel and Stub files one directory holds, by size and name: the two
    /// halves of the match, one loaded by the app and one by the DSP.
    private static func named(_ directory: String) -> [String] {
      shell("ls -l '\(directory)' 2>/dev/null | grep -iE 'skel|stub' | awk '{print $5, $NF}'")
    }

    /// One numbered line per directory: a path that does not exist is one the
    /// Hexagon cannot be handed a Skel from, and the numbering is the order.
    private static func skelSearch(_ search: [(directory: String, files: [String])]) -> [String] {
      guard !search.isEmpty else { return ["ADSP_LIBRARY_PATH is empty: the DSP has nowhere to look for a Skel"] }
      return search.enumerated().map { index, entry in
        "\(index + 1). \(entry.directory): \(entry.files.isEmpty ? "no Skel, no Stub" : entry.files.joined(separator: " / "))"
      }
    }

    /// The pair this run hands QNN, and the phone's own pair for the same
    /// architecture: the Skel the DSP will find first, the Stub the app will
    /// load, and what the phone carries instead. Two QNN builds in one process
    /// take no node of any model, so which pair is in force is the first thing
    /// to read here, and the phone's own is what a build that carries no pair
    /// for this Hexagon should have been using.
    private static func pairInForce(_ search: [(directory: String, files: [String])], arch: String?) -> [String] {
      var lines: [String] = []
      lines.append("the QNN backend this run loads: \(qnnBackend ?? "the app's own libQnnHtp.so, beside the app")")
      lines.append("the Phone's Own QNN setting this run: \(phoneOwnQnn ? "on" : "off")")
      if !phoneLibraries.isEmpty {
        lines.append("opened by path before the session, so every name finds these: \(phoneLibraries.joined(separator: " "))")
      }
      let held = search.first { $0.files.contains { $0.lowercased().contains("skel.so") } }
      if let held, let file = held.files.first(where: { $0.lowercased().contains("skel.so") }) {
        lines.append("the DSP looks for its Skel in \(held.directory) first, and finds: \(file)")
        if let native = nativeDirectory, held.directory == native {
          lines.append(
            "and that is the app's own copy, which no vendor signed: a DSP that will not take it takes no node of any model")
        }
      } else {
        lines.append("no directory in ADSP_LIBRARY_PATH holds a Skel: the DSP has none to load")
      }
      guard let arch else {
        lines.append("this build carries no HTP Stub, so no architecture's pair is in question")
        return lines
      }
      // The Stub the app's own directory holds: by name, the pair is one build
      // from one QAIRT release, and a Stub from another speaks another dialect.
      let stub = qnnFiles.first { $0.contains("\(arch)Stub") } ?? "libQnnHtp\(arch)Stub.so"
      lines.append("the app's Stub for \(arch): \(stub), the Skel it came with beside it")
      let own = ownDirectories().flatMap { listing($0) }.filter { $0.contains("libQnnHtp\(arch)") }
      if own.isEmpty {
        lines.append("the phone has no \(arch) pair of its own in any of: \(ownDirectories().joined(separator: " "))")
      } else {
        lines.append(contentsOf: own)
      }
      return lines
    }

    /// Where a phone's own QNN libraries have been seen. The DSP directories
    /// are the ones a search that stops at `/vendor/lib64` misses: a Skel is
    /// read by the Hexagon, so a vendor keeps it in an `rfsa` directory beside
    /// the Hexagon's own, and the phone this was written for keeps its whole
    /// V85 set under `/odm` -- backend and Stub in `/odm/lib64`, the Skel in
    /// `/odm/lib/rfsa/adsp`.
    private static func ownDirectories() -> [String] {
      [
        "/vendor/lib64", "/system/lib64", "/odm/lib64", "/vendor/lib",
        "/odm/lib/rfsa/adsp", "/vendor/lib/rfsa/adsp", "/vendor/lib/rfsa/cdsp", "/system/vendor/lib/rfsa/adsp",
      ]
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

    /// A title and the lines under it, or the one line that says it is empty. A
    /// body longer than `cap` is cut and says how many lines it left: a section
    /// that grew must never be the reason one below it was truncated, and the
    /// lines a reason lives in are the ones this report keeps.
    private static func section(_ title: String, _ body: [String], cap: Int = 24) -> [String] {
      guard !body.isEmpty else { return ["\(title): nothing"] }
      var lines = ["\(title):"] + body.prefix(cap).map { "  \($0)" }
      if body.count > cap {
        lines.append("  ... and \(body.count - cap) more")
      }
      return lines
    }

    /// The lines of `lines` that say one of `words`, in the order they came.
    private static func matching(_ lines: [String], _ words: [String]) -> [String] {
      lines.filter { line in
        let lower = line.lowercased()
        return words.contains { lower.contains($0) }
      }
    }

    /// One `ls -l` of a directory, as lines, or its own line saying why not.
    /// Every file line carries the directory it was read from: a phone holds
    /// three QNN builds in three directories, and a bare file name does not say
    /// which of them a file belongs to -- which is what a report is asked. The
    /// listing is kept for `spacing`, because the files in a directory do not
    /// change between the two reports one failed start writes; the listing is
    /// never the reason a report goes unwritten.
    private static func listing(_ path: String) -> [String] {
      lock.lock()
      let kept = listings[path]
      lock.unlock()
      if let kept, Date().timeIntervalSince(kept.at) < spacing { return kept.lines }
      let lines = shell("ls -l '\(path)'")
        .filter { !$0.hasPrefix("total ") }
        .map { $0.hasPrefix("ls:") ? $0 : "\(path)  \($0)" }
      lock.lock()
      listings[path] = (Date(), lines)
      lock.unlock()
      return lines
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
