import Foundation
import JetlinkKit

/// Feed the loaded engine moving synthetic frames and report whether its
/// pose/plan outputs actually move. This is the on-device twin of
/// workspace/diag1116/synth_probe.py: it bypasses the camera, the comma and
/// the wire, and drives the engine through the same stage/run/keep path a
/// real frame takes. The point is to separate "the QNN path computes a
/// frozen answer" from "the link or the camera never delivered motion".
///
/// A healthy model returns a plan that grows with the sliding texture and a
/// transx that tracks it (std of order 0.7 over 40 frames). A broken one
/// returns the same few numbers every frame (unique ~1, std ~0.005). The
/// numbers go to the server log, which the app's Share Logs button carries
/// out, so this can be run on a phone in someone else's hands.
extension EngineHost {
  /// One probe report, logged as a single WARNING line prefixed with
  /// `probe_numerics` so it survives a log tail and is easy to grep.
  @discardableResult
  public func probeNumerics(frames requested: Int) throws -> [String] {
    let frames = min(max(requested, 4), 240)
    // Own the engine exactly like the benchmark does. QNN HTP engines report
    // coolsWhenIdle, so tick() calls warmIfIdle() whenever nothing else is
    // using the engine; without these guards it races this probe on the same
    // session and traps the process (Swift exclusive-access / native QNN).
    // benchmarking=true makes warmIfIdle and the live frame path both stand
    // down; the host lock serialises every stage/run/reset here.
    lock.lock()
    defer { lock.unlock() }
    guard let l = loaded else {
      throw HostError.invalid("no model is loaded; prepare one first")
    }
    guard !benchmarking else {
      throw HostError.invalid("the engine is busy (benchmark or probe running)")
    }
    benchmarking = true
    defer { benchmarking = false }
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

    // A road-ish frame: two converging lines over a textured band, shifted
    // by the frame index so consecutive calls differ. Deliberately synthetic
    // and identical on every host; nonsense as a road, but the same nonsense.
    func frame(_ step: Int) -> [UInt8] {
      var img = [UInt8](repeating: 90, count: planeBytes)  // flat sky
      let horizon = h / 3
      var seed = UInt64(7)
      for y in horizon..<h {
        for x in 0..<w {
          seed = seed &* 6364136223846793005 &+ 1442695040888963407
          var v = UInt8(truncatingIfNeeded: (seed >> 33) & 0xff)
          v = 40 + min(v, 175)
          img[y * w + x] = v
        }
      }
      for centre in [w / 3 + step % 17, 2 * w / 3 + step % 17] {
        for y in horizon..<h {
          let cx = centre + (y - horizon) / 6
          let half = 2 + (y > horizon ? 2 : 0)
          let lo = max(0, cx - half), hi = min(w - 1, cx + half)
          if lo <= hi {
            for x in lo...hi { img[y * w + x] = 235 }
          }
        }
      }
      return img
    }

    var transx: [Double] = []
    var rotz: [Double] = []
    var planmax: [Double] = []

    guard let out = l.staging.layout.output else {
      throw HostError.failed("the engine has no driving output buffer")
    }
    l.staging.reset()
    for step in 0..<frames {
      // Only the newest plane matters for the output; fill the queue slots
      // with successive frames so a stateful graph sees motion too.
      var stack = [UInt8]()
      stack.reserveCapacity(slots * 6 * planeBytes)
      for slot in 0..<slots {
        let plane = frame(step + 4 * slot)
        for _ in 0..<6 { stack.append(contentsOf: plane) }
      }
      try stack.withUnsafeBytes { raw in
        try packed.withUnsafeBytes { p in
          try l.staging.stage(warped: raw.baseAddress!, packed: p.baseAddress!)
        }
      }
      try l.engine.run()

      output.withUnsafeMutableBytes { o in
        if io.type == .float16 {
          Convert.f16ToF32(out, o.baseAddress!, count: spec.outputCount)
        } else {
          o.baseAddress!.copyMemory(from: out, byteCount: spec.outputCount * 4)
        }
      }
      let pose = output[poseSlice]
      let plan = output[planSlice]
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
    }

    func summarize(_ name: String, _ xs: [Double]) -> String {
      guard !xs.isEmpty else { return "\(name) n=0" }
      let uniq = Set(xs.map { $0.rounded(to: 4) }).count
      let mean = xs.reduce(0, +) / Double(xs.count)
      let variance = max(0, xs.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Double(xs.count))
      return String(
        format: "%@ n=%d unique=%d std=%.5f min=%+.4f max=%+.4f",
        name, xs.count, uniq, variance.squareRoot(), xs.min()!, xs.max()!)
    }
    let device = backend.describe()["device"] ?? "?"
    let header = "device \(device), frames \(frames), \(EngineHost.buildLine(l.engine))"
    let lines = [
      header,
      summarize("planmax", planmax),
      summarize("rotz   ", rotz),
      summarize("transx ", transx),
      "HEALTHY: planmax 3-15 m with unique near frame count; FROZEN: planmax ~0.3 m with unique near 1",
    ]
    for line in lines { Log.write(.warning, "probe_numerics", line) }
    return lines
  }

  /// Name used by the command dispatcher in ServerController.
  public func probeNumericsAndSummarize(frames: Int) throws -> [String] {
    try probeNumerics(frames: frames)
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
