import Foundation

/// A value tests move by hand, such as a clock.
final class Dial<Value: Sendable>: @unchecked Sendable {
  private let lock = NSLock()
  private var stored: Value

  init(_ value: Value) {
    stored = value
  }

  var value: Value {
    get { lock.withLock { stored } }
    set { lock.withLock { stored = newValue } }
  }
}

/// Polls `condition` until it holds or `timeout` passes.
func eventually(timeout: TimeInterval = 5, _ condition: () -> Bool) -> Bool {
  let end = Date().addingTimeInterval(timeout)
  while Date() < end {
    if condition() { return true }
    Thread.sleep(forTimeInterval: 0.005)
  }
  return condition()
}
