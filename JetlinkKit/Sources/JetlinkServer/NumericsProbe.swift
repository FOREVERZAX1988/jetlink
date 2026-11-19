import Foundation
import JetlinkKit

/// A sheet of multi-scale texture, and the window of it the probe shows the
/// model each frame. The window shrinks -- the scene scales up around its
/// centre -- the way a forward-facing camera's view does as the car moves, so
/// the model's temporal half has a motion it can actually measure, unlike the
/// random bytes the benchmark feeds.
///
/// Pixels are picked nearest-neighbour: at these step sizes the flow, not the
/// antialiasing, is what the probe is asking about.
private struct ZoomingScene {
  let width: Int
  let height: Int
  let sheet: [UInt8]

  init(width: Int, height: Int) {
    self.width = width
    self.height = height
    var seed: UInt64 = 0x9E37_79B9_7F4A_7C15
    func random() -> Double {
      seed = seed &* 6364136223846793005 &+ 1442695040888963407
      return Double((seed >> 40) & 0xFFFF) / 65535.0
    }
    // Three octaves: blobs the window sees whole, mid detail, fine grain so the
    // flanks are never flat. Bilinear inside each octave keeps it smooth, which
    // is what makes a sub-pixel-per-frame expansion readable at all.
    let octaves: [(cw: Int, ch: Int, weight: Double)] = [(4, 3, 0.50), (10, 6, 0.30), (26, 14, 0.20)]
    var grids: [(data: [Double], cw: Int, ch: Int, weight: Double)] = []
    for octave in octaves {
      var data = [Double](repeating: 0, count: (octave.cw + 1) * (octave.ch + 1))
      for i in 0..<data.count { data[i] = random() }
      grids.append((data, octave.cw, octave.ch, octave.weight))
    }
    var pixels = [UInt8](repeating: 0, count: width * height)
    let spanX = Double(max(width - 1, 1))
    let spanY = Double(max(height - 1, 1))
    for y in 0..<height {
      for x in 0..<width {
        var value = 0.0
        for g in grids {
          let fx = Double(x) / spanX * Double(g.cw)
          let fy = Double(y) / spanY * Double(g.ch)
          let x0 = min(Int(fx), g.cw - 1)
          let y0 = min(Int(fy), g.ch - 1)
          let tx = fx - Double(x0)
          let ty = fy - Double(y0)
          let i = y0 * (g.cw + 1) + x0
          let top = g.data[i] * (1 - tx) + g.data[i + 1] * tx
          let bottom = g.data[i + g.cw + 1] * (1 - tx) + g.data[i + g.cw + 2] * tx
          value += g.weight * (top * (1 - ty) + bottom * ty)
        }
        pixels[y * width + x] = UInt8(clamping: Int((40 + value * 190).rounded()))
      }
    }
    sheet = pixels
  }

  /// One h*w plane: the centre of the sheet, seen at `scale`.
  func fill(scale: Double, into dest: UnsafeMutableRawPointer, w: Int, h: Int) {
    let stepX = 1.0 / max(scale, 1e-6)
    let stepY = 1.0 / max(scale, 1e-6)
    let cx = Double(width) * 0.5
    let cy = Double(height) * 0.5
    let left = cx - Double(w) * 0.5 * stepX
    let top = cy - Double(h) * 0.5 * stepY
    sheet.withUnsafeBytes { raw in
      guard let base = raw.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
      let out = dest.assumingMemoryBound(to: UInt8.self)
      for row in 0..<h {
        let sy = min(max(Int((top + Double(row) * stepY).rounded()), 0), height - 1)
        let src = base + sy * width
        let dst = out + row * w
        var sx = left
        for column in 0..<w {
          let x = min(max(Int(sx.rounded()), 0), width - 1)
          dst[column] = src[x]
          sx += stepX
        }
      }
    }
  }
}

/// Feed the loaded engine a scene that moves the way a road does and report
/// whether its pose/plan answers the motion. This is the on-device twin of
/// workspace/diag1116/synth_probe.py: it bypasses the camera, the comma and the
/// wire, and drives the engine through the same stage/run/keep path a real
/// frame takes.
///
/// The run is split in two halves with the scene expanding three times faster in
/// the second. A model whose temporal half works must answer that step: the
/// pose's forward velocity and the planned path have to grow with it. One whose
/// five-frame state never reaches the graph cannot see the rate at all, and
/// returns the same few numbers in both halves -- which is the difference
/// between "the frames are wrong" and "the state loop does not work here".
///
/// The summary goes to the server log and to the app's Share Logs; breadcrumbs
/// also go to `trace`, a file beside the app's cache that survives a crash, so
/// a trap is locatable after a restart.
///
/// Owns the engine exactly like `benchmark` does: the `benchmarking` flag keeps
/// `warmIfIdle` and the live frame path off the engine, and the host lock
/// covers each frame rather than the whole run.
extension EngineHost {
  @discardableResult
  public func probeNumerics(frames requested: Int, trace traceURL: URL? = nil) throws -> [String] {
    // Two halves of at least four frames each.
    let frames = min(max(requested, 8), 240)
    let half = frames / 2
    // Fractional expansion per frame. Slow is a lazy drive, fast is three
    // times that; the ratio is what the model has to reproduce.
    let slowRate = 0.010
    let fastRate = 0.030
    // The pose's first warmup frames are the state filling up, not an answer.
    let warmup = 4

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

    note("START frames=\(frames) slow=\(slowRate) fast=\(fastRate)")
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
    guard shape.count >= 4, shape[2] > 1, shape[3] > 1 else {
      throw HostError.failed("the warped shape is not [slots, 6, h, w]")
    }
    let slots = shape[0]
    let h = shape[2]
    let w = shape[3]
    let planeBytes = h * w
    let packed = [Float](repeating: 0, count: spec.packedCount)
    var output = [Float](repeating: 0, count: spec.outputCount)

    // The sheet is twice the frame in each direction, so the window can shrink
    // to a quarter before it runs out of texture -- more than the whole run
    // expands by.
    let scene = ZoomingScene(width: w * 3, height: h * 3)
    var stack = [UInt8](repeating: 128, count: slots * 6 * planeBytes)

    var transx: [Double] = []
    var rotz: [Double] = []
    var pathx: [Double] = []

    lock.lock()
    if loaded !== l {
      lock.unlock()
      throw HostError.failed("the engine was unloaded before the probe started")
    }
    l.staging.reset()
    lock.unlock()

    note("RESET ok warped=\(spec.warpedBytes) output=\(spec.outputCount) scene=\(scene.width)x\(scene.height)")
    // The comma's pace, as the benchmark keeps it: back-to-back runs are not
    // what the engine sees on a drive.
    let period = 1.0 / Double(ModelConstants.runFrequency)
    var next = ProcessInfo.processInfo.systemUptime
    var scale = 1.0
    for step in 0..<frames {
      let detailed = step == 0
      next += period
      let wait = next - ProcessInfo.processInfo.systemUptime
      if wait > 0 { Thread.sleep(forTimeInterval: wait) }

      // Additive, not geometric: the pixel flow per frame is what the model
      // measures, so it has to stay constant inside a half and triple at the
      // step. A geometric expansion would make the fast half flow hardest at
      // its end and read as more than the three times it is.
      let rate = step < half ? slowRate : fastRate
      scale += rate
      // No String(format:) here: %@ traps on Android's corelibs Foundation.
      if step % 5 == 0 { note("frame \(step)/\(frames) scale_milli=\(Int((scale * 1000).rounded()))") }

      // [slot][6][h][w]: slot 0 is the newest frame; inside a slot the first
      // three planes are one camera's YUV and the last three the other's. Older
      // slots saw a slightly smaller scene, which is the whole point.
      stack.withUnsafeMutableBytes { raw in
        let p = raw.baseAddress!
        for slot in 0..<slots {
          // The older slots saw the scene one frame's worth smaller: that
          // difference between them is the motion the graph has to read.
          let slotScale = scale - rate * Double(slot)
          let base = slot * 6 * planeBytes
          scene.fill(scale: slotScale, into: p + base, w: w, h: h)
          scene.fill(scale: slotScale, into: p + base + 3 * planeBytes, w: w, h: h)
          // Colourless chroma: the plane is flat 128, so only luma carries the
          // motion and nothing pretends to know a hue.
          for plane in [1, 2, 4, 5] {
            let target = p + base + plane * planeBytes
            target.assumingMemoryBound(to: UInt8.self).initialize(repeating: 128, count: planeBytes)
          }
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
      // An ArraySlice keeps the parent array's indices: output[poseSlice]
      // starts at 887, so pose[0] is out of bounds and traps the process with
      // no stack at all. Read from the slice's own first index instead.
      if pose.count >= 6 {
        let i = pose.startIndex
        transx.append(Double(pose[i]))
        rotz.append(Double(pose[i + 5]))
      }
      if detailed { note("f0 pose n=\(transx.count)") }
      // The path the model plans: position x at the last of the 33 steps, out
      // of the mean half of the 2x33x15 MDN block. That is metres, unlike the
      // raw tensor max, which is a mix of position and velocity.
      let lastStepX = 32 * 15
      if plan.count >= 2 * 33 * 15 {
        pathx.append(Double(plan[plan.startIndex + lastStepX]))
      }
      if detailed { note("f0 plan n=\(pathx.count)") }
      let finite = output.withUnsafeBytes { Convert.allFinite($0.baseAddress!, count: spec.outputCount) }
      if detailed { note("f0 finite=\(finite)") }
      // No keep(): a stateful graph keeps its own state, and its `keep` is the
      // empty one. A queued graph is not what these models ship as.
      if detailed { note("f0 done") }
      lock.unlock()
    }
    note("LOOP done n=\(transx.count)")

    func fixed(_ x: Double, _ digits: Int) -> String {
      guard x.isFinite else { return String(x) }  // nan / inf, never trap
      let scale = pow(10.0, Double(digits))
      return String(((x * scale).rounded()) / scale)
    }
    func mean(_ xs: [Double]) -> Double {
      xs.isEmpty ? 0 : xs.reduce(0, +) / Double(xs.count)
    }
    func spread(_ xs: [Double]) -> Double {
      guard !xs.isEmpty else { return 0 }
      let m = mean(xs)
      return max(0, xs.reduce(0) { $0 + ($1 - m) * ($1 - m) } / Double(xs.count)).squareRoot()
    }
    func unique(_ xs: [Double]) -> Int { Set(xs.map { $0.rounded(to: 4) }).count }

    /// The second half of a series, minus the warmup frames of that half.
    func phase(_ xs: [Double], _ second: Bool) -> [Double] {
      let lo = second ? half : 0
      let hi = second ? xs.count : min(half, xs.count)
      guard hi > lo + warmup else { return [] }
      return Array(xs[(lo + warmup)..<hi])
    }

    let slowTx = phase(transx, false)
    let fastTx = phase(transx, true)
    let slowPath = phase(pathx, false)
    let fastPath = phase(pathx, true)
    let ratio = mean(slowTx) != 0 ? mean(fastTx) / mean(slowTx) : 0

    let device = backend.describe()["device"] ?? "?"
    var lines = [
      "device \(device), frames \(frames), \(EngineHost.buildLine(l.engine))",
      "slow transx n=\(slowTx.count) mean=\(fixed(mean(slowTx), 4)) std=\(fixed(spread(slowTx), 4)) unique=\(unique(slowTx))",
      "fast transx n=\(fastTx.count) mean=\(fixed(mean(fastTx), 4)) std=\(fixed(spread(fastTx), 4)) unique=\(unique(fastTx))",
      "rotz all n=\(rotz.count) mean=\(fixed(mean(rotz), 5)) std=\(fixed(spread(rotz), 5)) unique=\(unique(rotz))",
      "slow pathx mean=\(fixed(mean(slowPath), 3)) fast pathx mean=\(fixed(mean(fastPath), 3))",
      "speed ratio fast/slow = \(fixed(ratio, 3))",
      "HEALTHY: ratio 2-4, both halves distinct every frame. DEAD: ratio ~1 with unique near 1 -- the rate never reaches the graph",
    ]
    // A short series so the shape of the answer is visible, not just its stats.
    var series: [String] = []
    var at = 0
    while at < transx.count {
      series.append("f\(at) tx=\(fixed(transx[at], 4)) px=\(fixed(at < pathx.count ? pathx[at] : 0, 2))")
      at += 6
    }
    if !series.isEmpty { lines.append("series " + series.joined(separator: " | ")) }
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
