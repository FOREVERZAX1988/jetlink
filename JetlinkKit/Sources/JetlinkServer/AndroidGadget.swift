import Foundation
import JetlinkKit

/// The comma's USB gadget as the Android app hands it over, and the server's
/// `GadgetSource` there. Android gives an app a USB device only through
/// UsbManager: the app finds the gadget by its IDs, asks for permission,
/// opens it, claims the vendor interface, and passes the file descriptor and
/// the two bulk endpoints' addresses to `attach`. The server opens pipes on
/// that descriptor for each session; it never closes it. When the device
/// goes, the app calls `detach` before closing the connection.
public final class AndroidGadget: @unchecked Sendable {
  public static let shared = AndroidGadget()

  private let lock = NSLock()
  private var attached: (device: UsbfsDevice, inEndpoint: UInt8, outEndpoint: UInt8, medium: LinkMedium?)?

  /// Takes over `fd`, the vendor interface already claimed, with its bulk IN
  /// and OUT endpoint addresses. Replaces any earlier device.
  public func attach(fd: Int32, inEndpoint: UInt8, outEndpoint: UInt8) throws {
    #if canImport(CUsbfs)
      let kernel = try LinuxUsbfs(fd: fd)
      let next = (UsbfsDevice(kernel: kernel), inEndpoint, outEndpoint, kernel.medium)
      lock.lock()
      let previous = attached
      attached = next
      lock.unlock()
      previous?.device.invalidate()
    #else
      throw LinkError.closed("this build has no usbfs")
    #endif
  }

  /// The device is going: ends every transfer on it and waits for them to
  /// let go of the descriptor, so the app can close it once this returns.
  public func detach() {
    lock.lock()
    let previous = attached
    attached = nil
    lock.unlock()
    previous?.device.invalidate()
  }

  /// The bus speed of the attached gadget, if one is attached.
  public var medium: LinkMedium? {
    lock.lock()
    defer { lock.unlock() }
    return attached?.medium
  }
}

extension AndroidGadget: GadgetSource {
  func present() -> Bool {
    lock.lock()
    defer { lock.unlock() }
    guard let attached else { return false }
    return !attached.device.isGone
  }

  func open() throws -> USBTransport {
    lock.lock()
    let current = attached
    lock.unlock()
    guard let current, !current.device.isGone else {
      throw LinkError.closed("no jetlink gadget is attached")
    }
    let pipes = UsbfsPipes(device: current.device, inEndpoint: current.inEndpoint, outEndpoint: current.outEndpoint)
    return USBTransport(pipes: pipes, medium: current.medium ?? .usb)
  }
}
