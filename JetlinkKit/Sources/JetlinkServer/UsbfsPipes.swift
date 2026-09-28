import Foundation
import JetlinkKit

#if canImport(CUsbfs)
  import CUsbfs
#endif

/// The comma's bulk pair through usbdevfs, Linux's user-space USB, on a file
/// descriptor someone else opened: on Android the app, through
/// UsbDeviceConnection. The Swift form of what libusb does for the Jetson's
/// Python server, and the Android counterpart of `IOUSBHostPipes`.
///
/// usbfs has one completion queue per file descriptor, and the kernel writes
/// a URB's status, length and any IN data only when the URB is reaped. So one
/// `UsbfsDevice` per descriptor owns the reaping: whichever transfer is
/// waiting polls the descriptor and reaps for everyone, and a transfer returns
/// only once its own URB is reaped. Nothing the kernel will write is freed
/// before then, and a session's pipes can close without leaving a URB behind
/// that a later session would reap into a freed buffer.
///
/// The kernel side is `UsbfsKernel`, so the tests run this against a fake on
/// any platform; `LinuxUsbfs` is the real one.
final class UsbfsPipes: BulkPipes, @unchecked Sendable {
  let device: UsbfsDevice
  let inEndpoint: UInt8
  let outEndpoint: UInt8

  init(device: UsbfsDevice, inEndpoint: UInt8, outEndpoint: UInt8) {
    self.device = device
    self.inEndpoint = inEndpoint
    self.outEndpoint = outEndpoint
  }

  func read(into buffer: UnsafeMutableRawPointer, count: Int, timeout: TimeInterval) throws -> Int {
    try device.transfer(endpoint: inEndpoint, buffer: buffer, count: count, timeout: timeout > 0 ? timeout : nil, owner: self)
  }

  func write(from buffer: UnsafeRawPointer, count: Int, timeout: TimeInterval) throws -> Int {
    try device.transfer(
      endpoint: outEndpoint, buffer: UnsafeMutableRawPointer(mutating: buffer), count: count, timeout: timeout > 0 ? timeout : nil, owner: self)
  }

  func abort() {
    device.cancel(owner: self)
  }

  /// Every transfer returns only once its URB is reaped, so closing is
  /// ending the ones in flight; the descriptor is the app's to close.
  func close() {
    abort()
  }

  deinit {
    device.forget(owner: self)
  }
}

/// One bulk transfer, from submit to reap.
final class UsbfsURB: @unchecked Sendable {
  let endpoint: UInt8
  let buffer: UnsafeMutableRawPointer
  let count: Int
  let owner: ObjectIdentifier
  /// Once reaped: 0, or the kernel's negative errno.
  var status: Int32 = 0
  var actual = 0
  var done = false
  /// The real kernel's URB while it is in flight.
  var handle: OpaquePointer?

  init(endpoint: UInt8, buffer: UnsafeMutableRawPointer, count: Int, owner: ObjectIdentifier) {
    self.endpoint = endpoint
    self.buffer = buffer
    self.count = count
    self.owner = owner
  }
}

/// usbfs as `UsbfsDevice` drives it. Calls come under the device's lock,
/// except `wait`, which one thread at a time makes without it, and `wake`.
protocol UsbfsKernel: AnyObject, Sendable {
  /// Queues `urb`: 0, or an errno.
  func submit(_ urb: UsbfsURB) -> Int32
  /// Ends `urb` early; it still has to be reaped.
  func discard(_ urb: UsbfsURB)
  /// A finished URB with its status and length filled in, or nil when none
  /// is ready. Throws once the device is gone and nothing is left to reap.
  func reap() throws -> UsbfsURB?
  /// Waits up to `timeout` (nil: forever) for a URB to reap or a `wake()`.
  /// False once the device is gone.
  func wait(timeout: TimeInterval?) -> Bool
  func wake()
}

/// Linux's errno values, which a URB's status carries whatever the platform
/// the code was built for: the fake kernel in the tests runs on a Mac.
enum LinuxErrno {
  static let noent: Int32 = 2
  static let again: Int32 = 11
  static let nomem: Int32 = 12
  static let nodev: Int32 = 19
  static let pipe: Int32 = 32
  static let proto: Int32 = 71
  static let overflow: Int32 = 75
  static let connreset: Int32 = 104
  static let shutdown: Int32 = 108

  static func describe(_ code: Int32) -> String {
    switch abs(code) {
    case noent, connreset: "cancelled"
    case nodev, shutdown: "the device is gone"
    case pipe: "the endpoint stalled"
    case proto: "a protocol error on the bus"
    case overflow: "the device sent more than was asked for"
    case nomem: "out of memory"
    default: "errno \(abs(code))"
    }
  }
}

/// One file descriptor's transfers, however many sessions' pipes use it.
final class UsbfsDevice: @unchecked Sendable {
  let kernel: any UsbfsKernel
  private let condition = NSCondition()
  /// A thread is in `kernel.wait` and reaps for everyone when it returns.
  private var polling = false
  private var gone = false
  private var inflight: [ObjectIdentifier: UsbfsURB] = [:]
  private var cancelled = Set<ObjectIdentifier>()
  /// Transfers between their first and last touch of the kernel.
  private var active = 0
  /// How long a discarded URB may take to come back before the wait for it
  /// polls again.
  static let discardPoll: TimeInterval = 0.05

  init(kernel: any UsbfsKernel) {
    self.kernel = kernel
  }

  var isGone: Bool {
    condition.lock()
    defer { condition.unlock() }
    return gone
  }

  /// Runs one transfer to its end and returns the bytes moved: all of them,
  /// or what arrived before `timeout` (nil: none) ended it, which is not an
  /// error. Throws when `cancel` ended it, the device went away, or the bus
  /// failed it.
  func transfer(endpoint: UInt8, buffer: UnsafeMutableRawPointer, count: Int, timeout: TimeInterval?, owner: AnyObject) throws -> Int {
    let id = ObjectIdentifier(owner)
    condition.lock()
    defer { condition.unlock() }
    if gone { throw LinkError.closed("the comma's USB device is gone") }
    if cancelled.contains(id) { throw LinkError.closed("the USB link was interrupted") }
    let urb = UsbfsURB(endpoint: endpoint, buffer: buffer, count: count, owner: id)
    let error = kernel.submit(urb)
    if error != 0 {
      throw LinkError.closed("could not queue a USB transfer: \(LinuxErrno.describe(error))")
    }
    inflight[ObjectIdentifier(urb)] = urb
    active += 1
    defer {
      active -= 1
      condition.broadcast()
    }
    let deadline = timeout.map { Date().addingTimeInterval($0) }
    var discarded = false
    while !urb.done {
      if gone {
        // Left in `inflight`, never freed: nothing reaps after this, so the
        // kernel never writes it, and the app's close of the descriptor ends it.
        throw LinkError.closed("the comma's USB device is gone")
      }
      if !discarded && (cancelled.contains(id) || deadline.map { Date() >= $0 } == true) {
        kernel.discard(urb)
        discarded = true
      }
      if polling {
        // Another transfer is in the kernel's wait and reaps for this one.
        if discarded {
          _ = condition.wait(until: Date().addingTimeInterval(UsbfsDevice.discardPoll))
        } else if let deadline {
          _ = condition.wait(until: deadline)
        } else {
          condition.wait()
        }
        continue
      }
      polling = true
      let limit: TimeInterval? = discarded ? UsbfsDevice.discardPoll : deadline.map { max(0, $0.timeIntervalSinceNow) }
      condition.unlock()
      let alive = kernel.wait(timeout: limit)
      condition.lock()
      polling = false
      reapAll()
      if !alive {
        gone = true
      }
      condition.broadcast()
    }
    if urb.status == 0 {
      return urb.actual
    }
    if urb.status == -LinuxErrno.noent || urb.status == -LinuxErrno.connreset {
      if cancelled.contains(id) {
        throw LinkError.closed("the USB link was interrupted")
      }
      if gone {
        throw LinkError.closed("the comma's USB device is gone")
      }
      // The deadline passed: what arrived is the caller's, as the pipes promise.
      return urb.actual
    }
    throw LinkError.closed("USB transfer on endpoint \(String(format: "%02x", endpoint)) failed: \(LinuxErrno.describe(urb.status))")
  }

  /// Under the lock: every finished URB marked done.
  private func reapAll() {
    while true {
      let urb: UsbfsURB?
      do {
        urb = try kernel.reap()
      } catch {
        gone = true
        return
      }
      guard let urb else { return }
      urb.done = true
      inflight[ObjectIdentifier(urb)] = nil
    }
  }

  /// Ends `owner`'s transfers, and any it starts later, with an error.
  func cancel(owner: AnyObject) {
    let id = ObjectIdentifier(owner)
    condition.lock()
    cancelled.insert(id)
    for urb in inflight.values where urb.owner == id {
      kernel.discard(urb)
    }
    condition.unlock()
    kernel.wake()
  }

  /// Forgets a closed owner, so a later one at the same address starts clean.
  func forget(owner: AnyObject) {
    condition.lock()
    cancelled.remove(ObjectIdentifier(owner))
    condition.unlock()
  }

  /// The device is going: every transfer ends with an error, and this waits
  /// up to `timeout` for them to leave the kernel alone, so the app can close
  /// the descriptor after. True when they all did.
  @discardableResult
  func invalidate(timeout: TimeInterval = 1.0) -> Bool {
    condition.lock()
    defer { condition.unlock() }
    gone = true
    for urb in inflight.values {
      kernel.discard(urb)
    }
    kernel.wake()
    condition.broadcast()
    let deadline = Date().addingTimeInterval(timeout)
    while active > 0 {
      if !condition.wait(until: deadline) {
        return active == 0
      }
    }
    return true
  }
}

#if canImport(CUsbfs)
  /// usbfs on a real file descriptor, which it does not own or close.
  final class LinuxUsbfs: UsbfsKernel, @unchecked Sendable {
    let fd: Int32
    private let wakeFD: Int32

    init(fd: Int32) throws {
      self.fd = fd
      wakeFD = jl_usbfs_wake_create()
      guard wakeFD >= 0 else { throw LinkError.closed("could not create the USB wake event") }
    }

    deinit {
      jl_usbfs_wake_close(wakeFD)
    }

    /// The bus speed as `USBDEVFS_GET_SPEED` reports it, or nil.
    var medium: LinkMedium? {
      switch jl_usbfs_speed(fd) {
      case 1, 2: .usb1
      case 3: .usb2
      case 5, 6: .usb3
      default: nil
      }
    }

    func submit(_ urb: UsbfsURB) -> Int32 {
      // The kernel keeps the URB alive until it is reaped.
      let context = Unmanaged.passRetained(urb).toOpaque()
      guard let handle = jl_urb_create(urb.endpoint, urb.buffer, Int32(urb.count), context) else {
        Unmanaged<UsbfsURB>.fromOpaque(context).release()
        return LinuxErrno.nomem
      }
      let error = jl_usbfs_submit(fd, handle)
      if error != 0 {
        Unmanaged<UsbfsURB>.fromOpaque(context).release()
        jl_urb_free(handle)
        return error
      }
      urb.handle = handle
      return 0
    }

    func discard(_ urb: UsbfsURB) {
      if let handle = urb.handle {
        _ = jl_usbfs_discard(fd, handle)
      }
    }

    func reap() throws -> UsbfsURB? {
      var done: OpaquePointer?
      let error = jl_usbfs_reap(fd, &done)
      if error == LinuxErrno.again { return nil }
      guard error == 0, let done, let context = jl_urb_context(done) else {
        throw LinkError.closed("USB reap failed: \(LinuxErrno.describe(error))")
      }
      let urb = Unmanaged<UsbfsURB>.fromOpaque(context).takeRetainedValue()
      urb.status = jl_urb_status(done)
      urb.actual = Int(jl_urb_actual(done))
      urb.handle = nil
      jl_urb_free(done)
      return urb
    }

    func wait(timeout: TimeInterval?) -> Bool {
      let ms = timeout.map { Int32(min(($0 * 1000).rounded(.up), Double(Int32.max))) } ?? -1
      return jl_usbfs_wait(fd, wakeFD, ms) != LinuxErrno.nodev
    }

    func wake() {
      jl_usbfs_wake(wakeFD)
    }
  }
#endif
