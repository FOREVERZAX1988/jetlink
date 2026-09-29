import Foundation
import JetlinkKit

/// The bulk IN pipe under a `ReadRing`: usbfs on Linux and Android, IOUSBHost
/// on a Mac. Every call is made under the ring's lock.
protocol ReadRingPipe: AnyObject {
  /// Posts a read of `size` bytes into `slot`'s buffer. Its completion comes
  /// back through `ReadRing.complete`. Throws when the pipe refuses it.
  func post(_ slot: Int, size: Int) throws
  /// Ends every read still posted, early. Each still completes, with what
  /// it had.
  func discardPosted()
  /// Waits until a read may have completed or `deadline` (nil: none) passes,
  /// releasing the ring's lock meanwhile. Throws once the link is gone or
  /// closed.
  func awaitCompletion(until deadline: Date?) throws
  /// Where `slot`'s bytes are.
  func bytes(_ slot: Int) -> UnsafeRawPointer
}

/// Reads kept posted ahead of the reader on the comma's bulk IN pipe, so the
/// comma's stream never waits for the host to ask for the next piece of a
/// message.
///
/// The host used to read a message in turns: a packet for the header, then up
/// to 256 KB, then the rest, each posted only once the one before completed.
/// The link idled through every turn, and with USB 3 link power management
/// on each turn also paid a U1/U2 exit: on a Jetson, 0.8 ms of the 2.1 ms
/// between a 475 KB request's header and its last byte. Here `depth` reads of
/// `slotSize` stay posted and are posted again as the reader empties them.
///
/// Why reading ahead is safe. The comma pads every message to
/// `Wire.gadgetTxAlign` (16 KB) and writes it in whole multiples of 16 KB
/// (`StreamTransport.send`, `FfsTransport.write_chunk`), so it never sends a
/// short packet and every message ends on the stream's 16 KB grid. A 16 KB
/// read posted on that grid ends at a message's end or inside the message,
/// never past it: it cannot sit holding the end of one message while it waits
/// for bytes of the next, which is why the host used to read no further than
/// the current message. Bytes go to the reader in the order the reads were
/// posted, whatever order they are reaped in, and never more than it asks
/// for, so a message still lands whole at the start of its buffer.
///
/// A short packet takes the stream off the grid, and a read posted after it
/// could straddle a message's end. The comma never sends one, so it means a
/// broken stream or a gadget that does not pad. The ring then ends every read
/// still posted, keeps what each got in order, and from then on posts only
/// what the reader asks for, as the host did before, until the link reopens.
///
/// One reader at a time. Every call is made under `lock`, which the pipe also
/// holds when it calls `complete`, and releases while it waits.
final class ReadRing: @unchecked Sendable {
  enum Completion {
    /// Bytes arrived, or none did (a zero-length packet).
    case data
    /// Ended early by `discardPosted`: what it got is still the stream's.
    case discarded
    case failed(LinkError)
  }

  private enum Slot {
    case idle, posted, done
  }

  static let slotSize = Wire.gadgetTxAlign
  /// 512 KB: a whole 400 KB inference request streams without the reader
  /// posting anything, with slots to spare.
  static let depth = 32

  let lock: NSCondition
  let depth: Int
  let slotSize: Int
  private unowned let pipe: any ReadRingPipe
  private var state: [Slot]
  private var sizes: [Int]
  private var actual: [Int]
  private var failures: [LinkError?]
  /// The oldest slot the reader has not emptied, and how much of it it took.
  private var head = 0
  private var offset = 0
  /// Slots from `head` on that are posted or done, in the order posted.
  private var used = 0
  /// Posted reads start on the 16 KB grid.
  private(set) var aligned: Bool
  /// A short read took the stream off the grid; the reader ends the reads
  /// posted after it.
  private var realign = false
  private let log = ServerLog(category: "usb")

  /// `aligned: false` posts only what the reader asks for, as a ring does
  /// after a short packet: the tests use it to compare the two.
  init(lock: NSCondition, pipe: any ReadRingPipe, depth: Int = ReadRing.depth, slotSize: Int = ReadRing.slotSize, aligned: Bool = true) {
    self.lock = lock
    self.pipe = pipe
    self.depth = depth
    self.slotSize = slotSize
    self.aligned = aligned
    state = Array(repeating: .idle, count: depth)
    sizes = Array(repeating: 0, count: depth)
    actual = Array(repeating: 0, count: depth)
    failures = Array(repeating: nil, count: depth)
  }

  /// Copies up to `count` bytes of the stream into `buffer` and returns how
  /// many: whatever has arrived, once anything has. Returns 0 when `deadline`
  /// (nil: none) passes first, leaving the reads posted, so nothing that
  /// arrives later is lost. Throws the first failed read's error, and again
  /// on every call after.
  func read(into buffer: UnsafeMutableRawPointer, count: Int, deadline: Date?) throws -> Int {
    guard count > 0 else { return 0 }
    while true {
      if realign {
        realign = false
        log.warning("a short packet from the comma mid-stream; reading message by message until the link reopens")
        pipe.discardPosted()
      }
      let copied = try take(into: buffer, count: count)
      if copied > 0 {
        return copied
      }
      // Emptied slots go back only now, when the reader would wait: the
      // read that ends a message returns without posting on its way out.
      try refill(count)
      if let deadline, Date() >= deadline {
        return 0
      }
      try pipe.awaitCompletion(until: deadline)
    }
  }

  /// Records that `slot`'s read ended with `count` bytes. The pipe wakes the
  /// reader after.
  func complete(_ slot: Int, _ completion: Completion, count: Int) {
    guard state[slot] == .posted else { return }
    state[slot] = .done
    switch completion {
    case .failed(let error):
      failures[slot] = error
      actual[slot] = 0
    case .data, .discarded:
      actual[slot] = min(max(count, 0), sizes[slot])
      if aligned && actual[slot] > 0 && actual[slot] < sizes[slot] {
        aligned = false
        realign = true
      }
    }
  }

  /// Empties done slots in the order they were posted, into `buffer`.
  private func take(into buffer: UnsafeMutableRawPointer, count: Int) throws -> Int {
    var copied = 0
    while copied < count && used > 0 && state[head] == .done {
      if let failure = failures[head] {
        throw failure
      }
      let n = min(actual[head] - offset, count - copied)
      if n > 0 {
        (buffer + copied).copyMemory(from: pipe.bytes(head) + offset, byteCount: n)
        copied += n
        offset += n
      }
      if offset == actual[head] {
        state[head] = .idle
        head = (head + 1) % depth
        used -= 1
        offset = 0
      }
    }
    return copied
  }

  /// On the grid, every free slot, 16 KB each. Off it, only once nothing is
  /// posted, and only the `count` bytes the reader asked for: the rest of what
  /// it is filling, which ends where the old one-read-at-a-time host ended.
  private func refill(_ count: Int) throws {
    if aligned {
      while used < depth {
        guard try post((head + used) % depth, size: slotSize) else { return }
      }
    } else if used == 0 {
      let packet = Pinned.usbMaxPacket
      var left = (count + packet - 1) / packet * packet
      while left > 0 && used < depth {
        let size = min(left, slotSize)
        guard try post((head + used) % depth, size: size) else { return }
        left -= size
      }
    }
  }

  /// False when the pipe refused a read but others are in hand, which the
  /// reader goes on with: the ring runs shallower until the next refill.
  private func post(_ slot: Int, size: Int) throws -> Bool {
    do {
      try pipe.post(slot, size: size)
    } catch {
      if used > 0 { return false }
      throw error
    }
    state[slot] = .posted
    sizes[slot] = size
    actual[slot] = 0
    failures[slot] = nil
    used += 1
    return true
  }
}
