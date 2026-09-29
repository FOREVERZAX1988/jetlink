import Foundation

/// Released once, waited on by any number of threads: the accept loop and
/// the dial loop can both wait for the same session to end.
final class Latch: @unchecked Sendable {
  private let condition = NSCondition()
  private var released = false

  func release() {
    condition.lock()
    released = true
    condition.broadcast()
    condition.unlock()
  }

  func wait() {
    condition.lock()
    while !released {
      condition.wait()
    }
    condition.unlock()
  }
}

/// A value behind a lock that escaping closures can share, which Mutex,
/// being noncopyable, cannot be.
package final class Locked<Value>: @unchecked Sendable {
  private let lock = NSLock()
  private var stored: Value

  package init(_ value: Value) {
    stored = value
  }

  package func withLock<R>(_ body: (inout Value) throws -> R) rethrows -> R {
    lock.lock()
    defer { lock.unlock() }
    return try body(&stored)
  }

  package var value: Value {
    get { withLock { $0 } }
    set { withLock { $0 = newValue } }
  }
}
