import Foundation

/// Compares two JSON values as JSONSerialization reads them. A key the Swift
/// side leaves out matches a Python null; a key only the Python sends is a
/// difference unless `ignored` names its path.
public struct JSONComparison {
  public let ignored: Set<String>
  public var differences: [String] = []

  public init(ignoring ignored: Set<String> = []) {
    self.ignored = ignored
  }

  public mutating func compare(python: Any, swift: Any, at path: String) {
    switch (python, swift) {
    case (let p as [String: Any], let s as [String: Any]):
      for (key, value) in p where s[key] == nil {
        let child = path.isEmpty ? key : "\(path)/\(key)"
        if !(value is NSNull) && !ignored.contains(child) && !ignored.contains("*/\(key)") {
          differences.append("only Python sends \(child)")
        }
      }
      for (key, value) in s {
        let child = path.isEmpty ? key : "\(path)/\(key)"
        guard let theirs = p[key] else {
          differences.append("only Swift has \(child)")
          continue
        }
        compare(python: theirs, swift: value, at: child)
      }
    case (let p as [Any], let s as [Any]):
      guard p.count == s.count else {
        differences.append("\(path): \(p.count) items in Python, \(s.count) in Swift")
        return
      }
      for (index, pair) in zip(p, s).enumerated() {
        compare(python: pair.0, swift: pair.1, at: "\(path)/\(index)")
      }
    default:
      if !JSONComparison.leavesEqual(python, swift) {
        differences.append("\(path): Python \(python), Swift \(swift)")
      }
    }
  }

  public static func leavesEqual(_ a: Any, _ b: Any) -> Bool {
    if a is NSNull || b is NSNull { return a is NSNull && b is NSNull }
    if let x = a as? String, let y = b as? String { return x == y }
    if let x = a as? NSNumber, let y = b as? NSNumber { return x.doubleValue == y.doubleValue }
    if let x = a as? Bool, let y = b as? Bool { return x == y }
    return false
  }
}
