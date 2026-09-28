import Foundation
import JetlinkTestSupport
import Testing

@testable import JetlinkKit

/// The lines a Python ControlServer wrote over a real Registry and cache
/// (JetlinkKit/Scripts/make_conformance_fixtures.py, `control`). Every field
/// the Swift decodes must be what the Python sent, and every field the Python
/// sends must be one the Swift reads or one named here as not needed.
struct PythonControlEventsTests {
  static let ignored: Set<String> = ["*/event", "*/t"]

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

  /// What the status page reads: `jsonLine` writes each line back as the
  /// Python did, nulls included, in any key order. It may add a null the
  /// Python left out (a waiting link's medium): a page reads both the same.
  @Test func jsonLineWritesWhatThePythonWrote() throws {
    for (number, line) in try Fixture.lines("python_control_events.jsonl").enumerated() {
      let python = try JSONSerialization.jsonObject(with: line) as! [String: Any]
      let t = (python["t"] as! NSNumber).doubleValue
      let written = try ControlEvent(jsonLine: line).jsonLine(at: Date(timeIntervalSince1970: t))
      #expect(written.last == 0x0A && written.dropLast().firstIndex(of: 0x0A) == nil, "line \(number + 1) is one line")
      let swift = try JSONSerialization.jsonObject(with: written)
      let (pythonKeys, swiftKeys) = (PythonControlEventsTests.keys(python), PythonControlEventsTests.keys(swift))
      let added = swiftKeys.subtracting(pythonKeys)
      #expect(pythonKeys.isSubset(of: swiftKeys), "line \(number + 1) lacks \(pythonKeys.subtracting(swiftKeys))")
      #expect(added.allSatisfy { PythonControlEventsTests.value(at: $0, in: swift) is NSNull }, "line \(number + 1) adds \(added)")
      var comparison = JSONComparison()
      comparison.compare(python: python, swift: swift, at: "")
      let differences = comparison.differences.filter { difference in !added.contains { difference == "only Swift has \($0)" } }
      #expect(differences.isEmpty, "line \(number + 1): \(differences)")
    }
  }

  @Test func jsonLineNamesTheEventsTheFixtureLacks() throws {
    let benchmark = ControlEvent.benchmark(BenchmarkEvent(state: "running", elapsed: 1, total: 60, frames: 20, frame: nil, report: nil, detail: ""))
    let object = try #require(try JSONSerialization.jsonObject(with: benchmark.jsonLine(at: Date(timeIntervalSince1970: 5))) as? [String: Any])
    #expect(object["event"] as? String == "benchmark" && (object["t"] as? NSNumber)?.doubleValue == 5)
    #expect(object["frame"] is NSNull && object["report"] is NSNull)
    let shutdown = try JSONSerialization.jsonObject(with: ControlEvent.shutdownRequest(ShutdownRequestEvent(reason: "car battery")).jsonLine())
    #expect((shutdown as? [String: Any])?["reason"] as? String == "car battery")
    let unknown = try JSONSerialization.jsonObject(with: ControlEvent.unknown(name: "later").jsonLine()) as? [String: Any]
    #expect(unknown?["event"] as? String == "later" && Set(unknown?.keys.map { $0 } ?? []) == ["event", "t"])
  }

  /// Every key path in a JSON value, as JSONComparison writes them.
  static func keys(_ value: Any, at path: String = "") -> Set<String> {
    func child(_ key: String) -> String { path.isEmpty ? key : "\(path)/\(key)" }
    switch value {
    case let object as [String: Any]:
      return object.reduce(into: Set<String>()) { out, pair in
        out.insert(child(pair.key))
        out.formUnion(keys(pair.value, at: child(pair.key)))
      }
    case let array as [Any]:
      return array.enumerated().reduce(into: Set<String>()) { out, pair in out.formUnion(keys(pair.element, at: child("\(pair.offset)"))) }
    default:
      return []
    }
  }

  static func value(at path: String, in root: Any) -> Any? {
    path.split(separator: "/").reduce(Optional(root)) { node, part in
      if let object = node as? [String: Any] { return object[String(part)] }
      if let array = node as? [Any], let index = Int(part), array.indices.contains(index) { return array[index] }
      return nil
    }
  }
}
