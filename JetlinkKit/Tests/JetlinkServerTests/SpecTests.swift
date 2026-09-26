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
    #expect(queued.packedShapes.map(\.name) == ["desire", "traffic_convention", "action_t", "prev_feat"])
    #expect(queued.packedCount == 8 + 2 + 2 + 32)
    #expect(queued.inferReqBytes == Wire.inferReqSize + 1536 + 4 * 44)

    let stateful = try ModelSpec.from(Fixture.json("tiny_stateful.spec.json"))
    #expect(stateful.stateful)
    #expect(Set(stateful.statePairs.map(\.input)) == ["state_img_q", "state_desire_q", "state_feat_q"])
    #expect(stateful.packedCount == 8 + 2 + 2)
  }
}
