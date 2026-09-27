import Foundation
import JetlinkTestSupport
import Testing

@testable import JetlinkKit

/// The lines a Python ControlServer wrote over a real Registry and cache
/// (JetlinkKit/Scripts/make_conformance_fixtures.py, `control`). Every field
/// the Swift decodes must be what the Python sent, and every field the Python
/// sends must be one the Swift reads or one named here as not needed.
struct PythonControlEventsTests {
  static let ignored: Set<String> = [
    "*/event", "*/t",
    // the stats event's older summaries; the apps draw served_ms and stages_ms
    "total_ms", "gpu_ms",
  ]

  private static func payload(_ event: ControlEvent) throws -> (String, Data)? {
    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    switch event {
    case .hello(let e): return ("hello", try encoder.encode(e))
    case .server(let e): return ("server", try encoder.encode(e))
    case .link(let e): return ("link", try encoder.encode(e))
    case .engine(let e): return ("engine", try encoder.encode(e))
    case .stats(let e): return ("stats", try encoder.encode(e))
    case .inventory(let e): return ("inventory", try encoder.encode(e))
    case .catalog(let e): return ("catalog", try encoder.encode(e))
    case .download(let e): return ("download", try encoder.encode(e))
    case .importEvent(let e): return ("import", try encoder.encode(e))
    case .benchmark(let e): return ("benchmark", try encoder.encode(e))
    case .shutdownRequest(let e): return ("shutdown_request", try encoder.encode(e))
    case .reply(let e): return ("reply", try encoder.encode(e))
    case .unknown: return nil
    }
  }

  @Test func theComparisonNoticesWhatItShould() throws {
    var comparison = JSONComparison(ignoring: ["*/t"])
    comparison.compare(
      python: ["t": 1.0, "a": 1, "b": ["x": "y", "z": NSNull()], "new": 2, "gone": NSNull()] as [String: Any],
      swift: ["a": 2, "b": ["x": "y"], "extra": true] as [String: Any], at: "")
    #expect(Set(comparison.differences) == ["a: Python 1, Swift 2", "only Python sends new", "only Swift has extra"])
  }

  @Test func theFixtureCoversEveryEventTheMacHears() throws {
    let lines = try Fixture.lines("python_control_events.jsonl")
    let names = Set(try lines.map { line in (try JSONSerialization.jsonObject(with: line) as! [String: Any])["event"] as! String })
    #expect(names == ["hello", "server", "link", "engine", "stats", "inventory", "catalog", "download", "import", "reply"])
  }

  @Test func everyLineDecodesToWhatThePythonWrote() throws {
    for (number, line) in try Fixture.lines("python_control_events.jsonl").enumerated() {
      let python = try JSONSerialization.jsonObject(with: line) as! [String: Any]
      let name = python["event"] as! String
      let event = try ControlEvent(jsonLine: line)
      guard let (decoded, data) = try PythonControlEventsTests.payload(event) else {
        Issue.record("line \(number + 1): \(name) decoded as unknown")
        continue
      }
      #expect(decoded == name, "line \(number + 1)")
      let swift = try JSONSerialization.jsonObject(with: data)
      var comparison = JSONComparison(ignoring: PythonControlEventsTests.ignored)
      comparison.compare(python: python, swift: swift, at: "")
      #expect(comparison.differences.isEmpty, "line \(number + 1) (\(name)): \(comparison.differences)")
    }
  }
}
