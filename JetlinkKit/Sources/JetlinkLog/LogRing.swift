import Foundation

/// The last lines this process logged, for the status page's /logs: the
/// journal has them too, but a page must not start journalctl. Linux's
/// `Logger` keeps every line it writes here; a Mac's daemon keeps its sink's.
public final class LogRing: @unchecked Sendable {
  public static let shared = LogRing(capacity: 300)

  public let capacity: Int
  private let lock = NSLock()
  private var times: [Date]
  private var texts: [String]
  private var next = 0
  private var count = 0

  public init(capacity: Int) {
    precondition(capacity > 0)
    self.capacity = capacity
    times = Array(repeating: Date(timeIntervalSince1970: 0), count: capacity)
    texts = Array(repeating: "", count: capacity)
  }

  /// Keeps `line`, dropping the oldest once full. The slots are written in
  /// place: a line costs a lock and no allocation on the thread that logs.
  public func append(_ line: String, at date: Date = Date()) {
    lock.lock()
    times[next] = date
    texts[next] = line
    next = (next + 1) % capacity
    count = min(count + 1, capacity)
    lock.unlock()
  }

  /// The kept lines, oldest first, each after its local time.
  public func lines() -> [String] {
    lock.lock()
    let first = (next - count + capacity) % capacity
    let kept = (0..<count).map { (times[(first + $0) % capacity], texts[(first + $0) % capacity]) }
    lock.unlock()
    return kept.map { "\(LogRing.timestamp($0)) \($1)" }
  }

  /// "2026-09-28 12:53:24.982" in local time: the journal's time is not on
  /// a page that reads the ring.
  static func timestamp(_ date: Date) -> String {
    var seconds = time_t(date.timeIntervalSince1970.rounded(.down))
    var parts = tm()
    localtime_r(&seconds, &parts)
    let millis = min(999, Int((date.timeIntervalSince1970 - Double(seconds)) * 1000))
    return String(
      format: "%04d-%02d-%02d %02d:%02d:%02d.%03d", Int(parts.tm_year) + 1900, Int(parts.tm_mon) + 1, Int(parts.tm_mday), Int(parts.tm_hour),
      Int(parts.tm_min), Int(parts.tm_sec), millis)
  }
}
