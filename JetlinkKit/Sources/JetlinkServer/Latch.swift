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
