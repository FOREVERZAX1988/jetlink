import Foundation
import Testing

@testable import JetlinkServer

@Suite("Model spec")
struct SpecTests {
  @Test("The spec read from the ONNX is the dict Python sends", arguments: ["tiny_queued", "tiny_stateful"])
  func matchesPython(_ name: String) throws {
    let python = try Fixture.json("\(name).spec.json")
    let spec = try ONNXPreparer().readSpec(
      model: Fixture.url("\(name).onnx"), sha256: python["sha256"] as! String,
      nbytes: (python["nbytes"] as! NSNumber).int64Value, frameSkip: 4)
    let swift = try JSONSerialization.jsonObject(with: JSONSerialization.data(withJSONObject: spec.dictionary())) as! NSDictionary
    #expect(swift == python as NSDictionary)
    // and back, as a comma or a sidecar would read it
    #expect(try ModelSpec.from(python).dictionary() as NSDictionary == spec.dictionary() as NSDictionary)
  }

  @Test("The wire sizes follow the model's shapes")
  func sizes() throws {
    let queued = try ModelSpec.from(Fixture.json("tiny_queued.spec.json"))
    #expect(!queued.stateful)
    #expect(queued.warpedShape == [2, 6, 8, 16])
    // prev_feat stays here (protocol 3), and so does hidden_state, 32 of the 64
    #expect(queued.packedShapes.map(\.name) == ["desire", "traffic_convention", "action_t"])
    #expect(queued.packedCount == 8 + 2 + 2)
    #expect(queued.prevFeatCount == 32)
    #expect(queued.inferReqBytes == Wire.inferReqSize + 1536 + 4 * 12)
    #expect(queued.hiddenRange == 32..<64)
    #expect(queued.inferRespBytes == Wire.inferRespSize + 4 * 32)

    let stateful = try ModelSpec.from(Fixture.json("tiny_stateful.spec.json"))
    #expect(stateful.stateful)
    #expect(Set(stateful.statePairs.map(\.input)) == ["state_img_q", "state_desire_q", "state_feat_q"])
    #expect(stateful.packedCount == 8 + 2 + 2)
    #expect(stateful.hiddenRange == 32..<48)
    #expect(stateful.inferRespBytes == Wire.inferRespSize + 4 * 48)
  }

  /// A 766 MB model's frame, whichever layout: one 16 KB read on the comma
  /// down (it was five), and 64 KB less up for a queued one.
  @Test("A big model's reply fits the one read the comma keeps posted")
  func bigModel() {
    let slices = [NamedRange("plan", 917..<1907), NamedRange("hidden_state", 2066..<18450), NamedRange("pad", 18450..<18452)]
    let queued = ModelSpec(
      sha256: "", nbytes: 0, frameSkip: 4,
      inputShapes: [
        NamedShape("img", [1, 12, 128, 256]), NamedShape("big_img", [1, 12, 128, 256]), NamedShape("desire_pulse", [1, 25, 8]),
        NamedShape("traffic_convention", [1, 2]), NamedShape("action_t", [1, 2]), NamedShape("features_buffer", [1, 32, 32, 512]),
      ], outputShapes: [NamedShape("outputs", [1, 18452])], outputSlices: slices, checkpoint: nil)
    #expect(Wire.headerSize + queued.inferReqBytes == 393_304)
    #expect(Wire.headerSize + queued.inferRespBytes == 8_324)
    #expect(queued.prevFeatCount == 16_384)
  }
}
