import JetlinkONNX

// Temporary: the converter's Report gains a summary of its own, and this
// file goes when the two branches meet.
extension LiteRTPreparation.Report {
  /// One log line: the rewrites that applied, the operators written, the size.
  public var summary: String {
    let applied = rewrites.filter { $0.value > 0 }.sorted { $0.key < $1.key }.map { "\($0.value) \($0.key)" }
    return "\(applied.joined(separator: ", ")); \(operators.values.reduce(0, +)) operators, "
      + "\(transposesRemoved) transposes removed, \(fileBytes / 1_000_000) MB"
  }
}
