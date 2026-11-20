import Foundation
import JetlinkKit

/// A real road, frame after frame, with no history of our own making.
///
/// Every other probe hands the graph five hand-built slots: it answers whether
/// the engine can read a rate when a perfect one is supplied. This one gives
/// it one real frame at a time and lets `next_state_*` become `state_*`, as a
/// live session does. A graph that answers one and not the other has a state
/// problem that no amount of camera work will fix.
extension EngineHost {
  @discardableResult
  public func replaySequence(file: URL, frames requested: Int, trace traceURL: URL? = nil) throws -> [String] {
    let data = try Data(contentsOf: file)
    let note = { (line: String) in
      Log.write(.warning, "probe_sequence", line)
      if let traceURL, let handle = try? FileHandle(forWritingTo: traceURL) {
        handle.seekToEndOfFile()
        handle.write(Data((line + "\n").utf8))
        try? handle.close()
      }
    }

    lock.lock()
    guard let l = loaded else {
      lock.unlock()
      throw HostError.failed("no model is loaded; prepare one first")
    }
    benchmarking = true
    lock.unlock()

    let shape = l.spec.warpedShape
    guard shape.count >= 4 else { throw HostError.failed("the warped shape is not [slots,6,h,w]") }
    let slots = shape[0]
    let h = shape[2]
    let w = shape[3]
    let plane = h * w
    let frameBytes = slots * 6 * plane  // all twelve planes, real YUV
    guard data.count % frameBytes == 0 else {
      throw HostError.invalid("\(file.lastPathComponent): \(data.count) bytes is not a whole number of \(frameBytes)-byte frames")
    }
    let frames = min(max(requested, 1), data.count / frameBytes)
    defer {
      lock.lock()
      benchmarking = false
      if loaded === l { l.staging.reset() }
      lock.unlock()
    }

    let spec = l.spec
    guard let io = l.engine.outputs[ModelConstants.drivingOutput] else {
      throw HostError.failed("the engine has no driving output")
    }
    guard let poseSlice = spec.outputSlices.first(where: { $0.name == "pose" })?.range(in: spec.outputCount),
      let planSlice = spec.outputSlices.first(where: { $0.name == "plan" })?.range(in: spec.outputCount)
    else {
      throw HostError.failed("the spec has no pose or plan slice")
    }
    let count = spec.outputCount
    var output = [Float](repeating: 0, count: count)
    var packed = [Float](repeating: 0, count: spec.packedCount)
    var stack = [UInt8](repeating: 128, count: spec.warpedBytes)

    note("START frames=\(frames) from \(file.lastPathComponent), engine loops its own state")

    var transx: [Double] = []
    var pathx: [Double] = []
    let period = 1.0 / Double(ModelConstants.runFrequency)
    var next = ProcessInfo.processInfo.systemUptime

    for step in 0..<frames {
      next += period
      let wait = next - ProcessInfo.processInfo.systemUptime
      if wait > 0 { Thread.sleep(forTimeInterval: wait) }

      // The full (2,6,h,w) frame, YUV included, copied byte for byte.
      data.withUnsafeBytes { raw in
        let base = raw.baseAddress!.assumingMemoryBound(to: UInt8.self)
        stack.withUnsafeMutableBytes { dest in
          dest.baseAddress!.copyMemory(from: base + step * frameBytes, byteCount: frameBytes)
        }
      }

      do {
        try stack.withUnsafeBytes { raw in
          try packed.withUnsafeMutableBytes { p in
            try l.staging.stage(warped: raw.baseAddress!, packed: p.baseAddress!)
          }
        }
      } catch {
        note("stage failed: \(String(describing: error))")
        throw error
      }
      do {
        try l.engine.run()
      } catch {
        note("run failed: \(String(describing: error))")
        throw error
      }
      if let out = l.staging.layout.output {
        output.withUnsafeMutableBytes { o in
          if io.type == .float16 {
            Convert.f16ToF32(out, o.baseAddress!, count: count)
          } else {
            o.baseAddress!.copyMemory(from: out, byteCount: count * 4)
          }
        }
      } else {
        throw HostError.failed("the engine has no driving output buffer")
      }

      let p = output[poseSlice]
      let plan = output[planSlice]
      guard p.count >= 6, plan.count >= 32 * 15 + 1 else { continue }
      let tx = Double(p[p.startIndex])
      let path = Double(plan[plan.startIndex + 32 * 15])
      transx.append(tx)
      pathx.append(path)
      if step % 10 == 0 {
        note("frame \(step)/\(frames) transx=\(tx) pathx=\(path)")
      }
    }

    func mean(_ xs: [Double]) -> Double { xs.isEmpty ? 0 : xs.reduce(0, +) / Double(xs.count) }
    let unique = Set(transx.map { Int(($0 * 1000).rounded()) }).count
    let lines = [
      "device \(backend.describe()["device"] ?? "?"), frames \(frames), source \(file.lastPathComponent), \(EngineHost.buildLine(l.engine))",
      "transx n=\(transx.count) mean=\(mean(transx)) min=\(transx.min() ?? 0) max=\(transx.max() ?? 0) unique=\(unique)",
      "pathx n=\(pathx.count) mean=\(mean(pathx)) min=\(pathx.min() ?? 0) max=\(pathx.max() ?? 0)",
      "ALIVE: transx varies over the road and is not pinned near 0. DEAD: transx ~ -0.07 as on the drive, with the frames moving.",
    ]
    for line in lines { note(line) }
    note("DONE ok=1")
    return lines
  }
}
