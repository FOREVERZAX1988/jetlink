import Foundation
import Testing

@testable import JetlinkServer

@Suite("Staging")
struct StagingTests {
  /// The session stages straight out of the receive buffer, where nothing
  /// promises the packed floats their alignment: staged one byte off it,
  /// every frame must still be what Python's queues feed.
  @Test("A frame is staged from wherever the request put it", arguments: [1, 4])
  func unaligned(frameSkip: Int) throws {
    let manifest = try Conformance.json("staging.json")
    let entry = try #require((manifest["cases"] as! [[String: Any]]).first { int($0["frame_skip"]) == frameSkip })
    let spec = try ModelSpec.from(Conformance.json(entry["spec"] as! String))
    let inputs = (entry["inputs"] as! [[String: Any]]).map {
      TensorSpec(name: $0["name"] as! String, type: .float16, shape: ($0["shape"] as! [NSNumber]).map(\.intValue))
    }
    let engine = StagingEngine(inputs)
    let staging = try PolicyQueues(spec: spec, engine: engine)
    let frames = try Conformance.data(entry["frames"] as! String)
    let staged = try Conformance.data(entry["staged"] as! String)
    let frameBytes = spec.warpedBytes + spec.packedBytes
    let stagedBytes = inputs.reduce(0) { $0 + $1.byteCount }
    let request = UnsafeMutableRawPointer.allocate(byteCount: frameBytes + 1, alignment: 16) + 1
    defer { (request - 1).deallocate() }
    for frame in 0..<int(manifest["frames"]) {
      if frame == int(manifest["reset_before"]) {
        staging.reset()
      }
      frames.withUnsafeBytes { request.copyMemory(from: $0.baseAddress! + frame * frameBytes, byteCount: frameBytes) }
      try staging.stage(warped: request, packed: request + spec.warpedBytes)
      var offset = frame * stagedBytes
      for input in inputs {
        let got = Data(bytes: engine.hostInput(input.name)!, count: input.byteCount)
        #expect(got == staged.subdata(in: offset..<offset + input.byteCount), "frame \(frame) \(input.name)")
        offset += input.byteCount
      }
    }
  }
}
