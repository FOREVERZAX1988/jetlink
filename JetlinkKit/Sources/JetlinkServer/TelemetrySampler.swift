import Foundation

/// The host's sensors, read off the frame path: `ServerHooks.telemetry` is
/// called on the session thread as a reply goes out, and a sensor read there
/// is sysfs or driver I/O that can stall. Python's `CachedTelemetry`.
///
/// One worker thread samples `source` when a reader wants a sample, at most
/// once a `period`. `read()` hands back the latest sample, never waits for a
/// new one, and gives `{}` once it is older than `maxAge`, which the comma
/// takes as "no health" rather than as a cold idle board. The worker never
/// holds the lock while it reads, so a stuck sensor costs stale telemetry,
/// never a frame.
public final class TelemetrySampler: @unchecked Sendable {
  public typealias Source = @Sendable () -> [String: Any]

  public let period: TimeInterval
  public let maxAge: TimeInterval
  private let source: Source
  private let clock: @Sendable () -> TimeInterval
  private let condition = NSCondition()
  private var sample: [String: Any] = [:]
  private var sampleTime = -TimeInterval.infinity
  private var nextRead: TimeInterval = 0
  /// Asked for at the start, so the first hello has a reading.
  private var requested = true
  private var closed = false

  public init(
    period: TimeInterval = 0.1, maxAge: TimeInterval = 1.0, clock: @escaping @Sendable () -> TimeInterval = { ProcessInfo.processInfo.systemUptime },
    source: @escaping Source
  ) {
    precondition(period > 0 && maxAge >= period, "require 0 < period <= maxAge")
    self.period = period
    self.maxAge = maxAge
    self.clock = clock
    self.source = source
    let thread = Thread { [self] in run() }
    thread.name = "jetlink-telemetry"
    thread.start()
  }

  /// The latest sample, or `{}` when there is none fresh enough. Asks the
  /// worker for the next one once a period has passed.
  public func read() -> [String: Any] {
    let now = clock()
    condition.lock()
    defer { condition.unlock() }
    if closed { return [:] }
    if now >= nextRead {
      requested = true
      condition.signal()
    }
    return now - sampleTime <= maxAge ? sample : [:]
  }

  /// Stops sampling. The worker is not waited for: it may be stuck in a
  /// sensor's driver, and it ends on its own once that returns.
  public func close() {
    condition.lock()
    closed = true
    condition.signal()
    condition.unlock()
  }

  private func run() {
    while true {
      condition.lock()
      while !requested && !closed {
        condition.wait()
      }
      if closed {
        condition.unlock()
        return
      }
      requested = false
      // The age starts before the read, or a slow read would look fresh.
      let started = clock()
      nextRead = .infinity
      condition.unlock()
      let fresh = source()
      condition.lock()
      sample = fresh
      sampleTime = started
      nextRead = clock() + period
      condition.unlock()
    }
  }
}
