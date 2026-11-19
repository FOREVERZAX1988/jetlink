import Foundation
import JetlinkKit

/// Feed the loaded engine moving synthetic frames and report whether its
/// pose/plan outputs actually move. This is the on-device twin of
/// workspace/diag1116/synth_probe.py: it bypasses the camera, the comma and
/// the wire, and drives the engine through the same stage/run/keep path a
/// real frame takes.
///
/// A healthy model returns a plan that grows with the sliding texture and a
/// transx that tracks it. A broken one returns the same few numbers every
/// frame (unique ~1, std ~0). The summary goes to the server log and to the
/// app's Share Logs; breadcrumbs also go to `trace`, a file beside the app's
/// cache that survives a crash, so a trap is locatable after a restart.
///
/// Owns the engine exactly like `benchmark` does: the `benchmarking` flag
/// keeps `warmIfIdle` and the live frame path off the engine, and the host
/// lock covers each frame rather than the whole run.
extension EngineHost {
  @discardableResult
  public func probeNumerics(frames requested: Int, trace traceURL: URL? = nil) throws -> [String] {
    let frames = min(max(requested, 4), 240)

    // Plain interpolation only: String(format:) with %@ segfaults on Android's
    // corelibs Foundation, and a trap here would take the whole app down.
    func note(_ line: String) {
      Log.write(.warning, "probe_numerics", line)
      guard let url = traceURL else { return }
      let data = (line + "\n").data(using: .utf8) ?? Data()
      if let handle = try? FileHandle(forWritingTo: url) {
        handle.seekToEndOfFile()
        handle.write(data)
        try? handle.close()
      } else {
        try? data.write(to: url)
      }
    }

    lock.lock()
    guard let l = loaded else {
      lock.unlock()
      throw HostError.invalid("no model is loaded; prepare one first")
    }
    guard !benchmarking else {
      lock.unlock()
      throw HostError.invalid("the engine is busy (benchmark or probe running)")
    }
    benchmarking = true
    lock.unlock()
    defer {
      lock.lock()
      benchmarking = false
      if loaded === l { l.staging.reset() }
      lock.unlock()
    }

    note("START frames=\(frames)")
    let spec = l.spec
    guard let io = l.engine.outputs[ModelConstants.drivingOutput],
      io.type == .float || io.type == .float16
    else {
      throw HostError.failed("the engine's driving output is not float")
    }
    guard let poseSlice = spec.outputSlices.first(where: { $0.name == "pose" })?.range(in: spec.outputCount),
      let planSlice = spec.outputSlices.first(where: { $0.name == "plan" })?.range(in: spec.outputCount)
    else {
      throw HostError.failed("the spec has no pose or plan slice")
    }

    let shape = spec.warpedShape  // [slots, 6, h, w]
    let slots = shape[0]
    let h = shape[2]
    let w = shape[3]
    let planeBytes = h * w
    let packed = [Float](repeating: 0, count: spec.packedCount)
    var output = [Float](repeating: 0, count: spec.outputCount)

    // Both buffers live outside the loop, as the benchmark's warped does: a
    // fresh 384 KB array every frame is 15 MB of churn, and a phone that the
    // system kills for it leaves no crash stack at all -- which is what the
    // probe kept looking like.
    var stack = [UInt8](repeating: 90, count: slots * 6 * planeBytes)
    var plane = [UInt8](repeating: 90, count: planeBytes)
    let horizon = h / 3

    // A road-ish frame: two converging lines over a textured band, shifted by
    // the frame index so consecutive calls differ. Deliberately synthetic and
    // identical on every host; nonsense as a road, but the same nonsense.
    func fillPlane(_ step: Int) {
      for i in 0..<planeBytes { plane[i] = 90 }  // flat sky
      // Seeded by the step so no two channels are byte-identical.
      var seed = UInt64(7) &+ UInt64(bitPattern: Int64(step)) &* 0x9E37_79B9_7F4A_7C15
      for y in horizon..<h {
        for x in 0..<w {
          seed = seed &* 6364136223846793005 &+ 1442695040888963407
          var v = UInt8(truncatingIfNeeded: (seed >> 33) & 0xff)
          v = 40 + min(v, 175)
          plane[y * w + x] = v
        }
      }
      for centre in [w / 3 + step % 17, 2 * w / 3 + step % 17] {
        for y in horizon..<h {
          let cx = centre + (y - horizon) / 6
          let half = 2 + (y > horizon ? 2 : 0)
          let lo = max(0, cx - half), hi = min(w - 1, cx + half)
          if lo <= hi {
            for x in lo...hi { plane[y * w + x] = 235 }
          }
        }
      }
    }

    var transx: [Double] = []
    var rotz: [Double] = []
    var planmax: [Double] = []

    lock.lock()
    if loaded !== l {
      lock.unlock()
      throw HostError.failed("the engine was unloaded before the probe started")
    }
    l.staging.reset()
    lock.unlock()

    note("RESET ok warped=\(spec.warpedBytes) output=\(spec.outputCount)")
    // The comma's pace, as the benchmark keeps it: back-to-back runs are not
    // what the engine sees on a drive.
    let period = 1.0 / Double(ModelConstants.runFrequency)
    var next = ProcessInfo.processInfo.systemUptime
    for step in 0..<frames {
      let detailed = step == 0
      next += period
      let wait = next - ProcessInfo.processInfo.systemUptime
      if wait > 0 { Thread.sleep(forTimeInterval: wait) }

      if step % 10 == 0 { note("frame \(step)/\(frames)") }
      // Only the newest plane matters for the output; fill the queue slots
      // with successive frames so a stateful graph sees motion too. Each of
      // the six channels gets its own content, as a real warped frame's do --
      // six copies of one plane is an input the model never sees on a drive.
      for slot in 0..<slots {
        let base = slot * 6 * planeBytes
        for k in 0..<6 {
          fillPlane(step + 4 * slot + k * 7)
          let off = base + k * planeBytes
          stack.replaceSubrange(off..<(off + planeBytes), with: plane)
        }
      }
      if detailed { note("f0 built") }
      lock.lock()
      guard loaded === l else {
        lock.unlock()
        throw HostError.failed("the engine was unloaded during the probe")
      }
      // Taken every frame, as the benchmark does: a stateful graph swaps its
      // output buffer between runs, so a cached pointer can go stale.
      guard let out = l.staging.layout.output else {
        lock.unlock()
        throw HostError.failed("the engine has no driving output buffer")
      }
      do {
        try stack.withUnsafeBytes { raw in
          try packed.withUnsafeBytes { p in
            try l.staging.stage(warped: raw.baseAddress!, packed: p.baseAddress!)
          }
        }
      } catch {
        lock.unlock()
        throw error
      }
      if detailed { note("f0 staged") }
      do {
        try l.engine.run()
      } catch {
        lock.unlock()
        throw error
      }
      if detailed { note("f0 ran") }
      output.withUnsafeMutableBytes { o in
        if io.type == .float16 {
          Convert.f16ToF32(out, o.baseAddress!, count: spec.outputCount)
        } else {
          o.baseAddress!.copyMemory(from: out, byteCount: spec.outputCount * 4)
        }
      }
      let pose = output[poseSlice]
      let plan = output[planSlice]
      if detailed { note("f0 read") }
      if pose.count >= 6 {
        transx.append(Double(pose[0]))
        rotz.append(Double(pose[5]))
      }
      if !plan.isEmpty { planmax.append(Double(plan.map { abs($0) }.max() ?? 0)) }
      // A queued graph feeds the prior frame's hidden state back via keep();
      // a stateful graph keeps its own state, so this is a no-op for it. Only
      // feed back finite output, matching the frame path and the benchmark.
      let finite = output.withUnsafeBytes { Convert.allFinite($0.baseAddress!, count: spec.outputCount) }
      if finite { l.staging.keep(outputs: out, type: io.type) }
      if detailed { note("f0 kept") }
      lock.unlock()
    }
    note("LOOP done n=\(transx.count)")

    func fixed(_ x: Double, _ digits: Int) -> String {
      guard x.isFinite else { return String(x) }  // nan / inf, never trap
      let scale = pow(10.0, Double(digits))
      return String(((x * scale).rounded()) / scale)
    }
    func summarize(_ name: String, _ xs: [Double]) -> String {
      guard !xs.isEmpty else { return "\(name) n=0" }
      let uniq = Set(xs.map { $0.rounded(to: 4) }).count
      let mean = xs.reduce(0, +) / Double(xs.count)
      let variance = max(0, xs.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Double(xs.count))
      let lo = xs.min() ?? 0, hi = xs.max() ?? 0
      return "\(name) n=\(xs.count) unique=\(uniq) std=\(fixed(variance.squareRoot(), 5))"
        + " min=\(fixed(lo, 4)) max=\(fixed(hi, 4))"
    }
    let device = backend.describe()["device"] ?? "?"
    let header = "device \(device), frames \(frames), \(EngineHost.buildLine(l.engine))"
    let lines = [
      header,
      summarize("planmax", planmax),
      summarize("rotz", rotz),
      summarize("transx", transx),
      "HEALTHY: planmax 3-15 m with unique near frame count; FROZEN: planmax ~0.3 m with unique near 1",
    ]
    for line in lines { note(line) }
    note("DONE ok=1")
    return lines
  }
}

extension Double {
  /// Round to `places` decimals for the unique-value count: float noise in
  /// the last bits must not make a frozen answer look alive.
  func rounded(to places: Int) -> Double {
    let scale = pow(10.0, Double(places))
    return (self * scale).rounded() / scale
  }
}
