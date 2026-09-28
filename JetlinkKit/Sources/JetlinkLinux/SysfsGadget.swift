#if os(Linux)
  import CUsbfs
  import Foundation
  import Glibc
  import JetlinkKit
  import JetlinkServer

  /// What `SysfsGadget` hands the claimed interface to: the server's
  /// `UsbfsGadget`, Android's transport, or a test's stand-in.
  protocol UsbfsTarget: GadgetSource {
    func attach(fd: Int32, inEndpoint: UInt8, outEndpoint: UInt8) throws
    func detach()
  }

  extension UsbfsGadget: UsbfsTarget {}

  /// The usbdevfs calls a claim makes, so the tests can stand in for the
  /// kernel. Each failure carries the errno.
  protocol UsbfsNode: Sendable {
    func open(_ path: String) throws(KernelError) -> Int32
    func descriptors(_ fd: Int32) throws(KernelError) -> [UInt8]
    /// The kernel driver bound to the interface, or nil.
    func driver(_ fd: Int32, interface: UInt8) throws(KernelError) -> String?
    func disconnect(_ fd: Int32, interface: UInt8) throws(KernelError)
    func claim(_ fd: Int32, interface: UInt8) throws(KernelError)
    func release(_ fd: Int32, interface: UInt8)
    func close(_ fd: Int32)
  }

  /// The comma's gadget on a Jetson or a Linux PC: found in sysfs, opened
  /// through usbdevfs, and served by the same `UsbfsGadget` Android's app
  /// feeds. Android's UsbManager finds, opens and claims for the app; here
  /// this does it (D7), with no libusb.
  ///
  /// `present()` is sysfs alone, as Python's was: a comma whose gadget cannot
  /// be claimed is still on the bus. `open()` claims on first sight and keeps
  /// the descriptor across the comma's per-run reopens, as on Android; the
  /// claim goes when the gadget leaves the bus, re-enumerates, or its
  /// transfers see ENODEV.
  public final class SysfsGadget: @unchecked Sendable {
    /// The gadget as sysfs names it.
    struct Found: Equatable {
      let name: String
      let bus: Int
      let device: Int
      /// Mb/s as sysfs says, for the log.
      let speed: String?
    }

    struct Claimed {
      let found: Found
      let fd: Int32
      let interface: UInt8
    }

    let root: HostRoot
    private let node: any UsbfsNode
    private let target: any UsbfsTarget
    private let log: LinuxLog
    private let lock = NSLock()
    private var claimed: Claimed?
    private var seen: Found?

    public convenience init() {
      self.init(root: .system, node: DevUsbfs(), target: UsbfsGadget())
    }

    init(root: HostRoot, node: any UsbfsNode, target: any UsbfsTarget, log: @escaping LinuxLog = serverLog("usb")) {
      self.root = root
      self.node = node
      self.target = target
      self.log = log
    }

    /// 1209:0001 in /sys/bus/usb/devices, the first by name.
    func scan() -> Found? {
      let devices = "/sys/bus/usb/devices"
      let vendor = String(format: "%04x", Pinned.usbVendorID)
      let product = String(format: "%04x", Pinned.usbProductID)
      for name in root.list(devices) where !name.contains(":") {
        let base = root.path("\(devices)/\(name)")
        guard Sysfs.read("\(base)/idVendor")?.lowercased() == vendor, Sysfs.read("\(base)/idProduct")?.lowercased() == product,
          let bus = Sysfs.readInt("\(base)/busnum"), let device = Sysfs.readInt("\(base)/devnum")
        else { continue }
        return Found(name: name, bus: bus, device: device, speed: Sysfs.read("\(base)/speed"))
      }
      return nil
    }

    /// The usbfs node sysfs's bus and device numbers name.
    func nodePath(_ found: Found) -> String {
      root.path(String(format: "/dev/bus/usb/%03d/%03d", found.bus, found.device))
    }

    /// Opens the node, finds the vendor interface, and claims it for `target`.
    private func claim(_ found: Found) throws -> Claimed {
      let path = nodePath(found)
      let fd = try node.open(path)
      do {
        let active = Sysfs.readInt(root.path("/sys/bus/usb/devices/\(found.name)/bConfigurationValue"))
        let picked = try GadgetDescriptors.pick(node.descriptors(fd), configuration: active)
        if let driver = try node.driver(fd, interface: picked.interface) {
          // "usbfs" is another program's claim, and taking the interface
          // from it would break that program mid-transfer: an old server
          // still running, most likely.
          guard driver != "usbfs" else {
            throw LinkError.closed("interface \(picked.interface) of \(path) is claimed by another program")
          }
          try node.disconnect(fd, interface: picked.interface)
          log(.info, "detached the \(driver) driver from the comma's interface \(picked.interface)")
        }
        try node.claim(fd, interface: picked.interface)
        do {
          try target.attach(fd: fd, inEndpoint: picked.inEndpoint, outEndpoint: picked.outEndpoint)
        } catch {
          node.release(fd, interface: picked.interface)
          throw error
        }
        // The link's power management goes here once it is measured: the
        // comma's port is the device's `port` link (2-1.3/port, which is
        // 2-1:1.0/2-1-port3), whose usb3_lpm_permit would be set now, before
        // the first session.
        log(
          .info,
          "claimed the comma's gadget at \(found.name) (\(path), \(found.speed.map { "\($0) Mb/s" } ?? "unknown speed")): interface \(picked.interface), bulk in \(hex(picked.inEndpoint)) out \(hex(picked.outEndpoint))"
        )
        return Claimed(found: found, fd: fd, interface: picked.interface)
      } catch {
        node.close(fd)
        throw error
      }
    }

    /// Under `lock`: stops the transfers, lets go of the interface, closes.
    private func releaseClaim(_ reason: String) {
      guard let claimed else { return }
      self.claimed = nil
      target.detach()
      node.release(claimed.fd, interface: claimed.interface)
      node.close(claimed.fd)
      log(.info, "released the comma's gadget at \(claimed.found.name): \(reason)")
    }

    private func hex(_ endpoint: UInt8) -> String {
      String(format: "%02x", endpoint)
    }
  }

  extension SysfsGadget: GadgetSource {
    public func present() -> Bool {
      let found = scan()
      lock.withLock {
        seen = found
        guard let claimed else { return }
        guard let found else {
          releaseClaim("it left the bus")
          return
        }
        if found != claimed.found {
          releaseClaim("it came back as \(found.name) device \(found.device)")
        } else if !target.present() {
          releaseClaim("the device is gone")
        }
      }
      return found != nil
    }

    public func open() throws -> any MessageLink {
      try lock.withLock {
        if claimed != nil && !target.present() {
          releaseClaim("the device is gone")
        }
        guard claimed == nil else { return }
        guard let found = seen ?? scan() else { throw LinkError.closed("no jetlink gadget is on the bus") }
        claimed = try claim(found)
      }
      return try target.open()
    }
  }

  /// The vendor interface and its bulk pair, read from the descriptors the
  /// gadget sends: FunctionFS numbers endpoints at bind, so they are never
  /// the same two constants.
  struct GadgetDescriptors: Equatable {
    let interface: UInt8
    let inEndpoint: UInt8
    let outEndpoint: UInt8

    private struct Interface {
      let number: UInt8
      let alternate: UInt8
      let kind: [UInt8]
      var endpoints: [(address: UInt8, attributes: UInt8)] = []
    }

    /// From a usbfs node's bytes (the device descriptor, then every
    /// configuration): in configuration `configuration` (bConfigurationValue;
    /// nil takes the first), the FF/FF/FF interface, else interface 0.
    static func pick(_ bytes: [UInt8], configuration: Int? = nil) throws -> GadgetDescriptors {
      var interfaces: [Interface] = []
      var inWanted = false
      var index = 0
      while index + 2 <= bytes.count {
        let length = Int(bytes[index])
        guard length >= 2, index + length <= bytes.count else { break }
        let field = { (offset: Int) in bytes[index + offset] }
        switch field(1) {
        case 2 where length >= 9:
          // A configuration: take this one, and stop at the next.
          if inWanted { index = bytes.count; continue }
          inWanted = configuration.map { Int(field(5)) == $0 } ?? true
        case 4 where length >= 9 && inWanted:
          interfaces.append(Interface(number: field(2), alternate: field(3), kind: [field(5), field(6), field(7)]))
        case 5 where length >= 7 && inWanted && !interfaces.isEmpty:
          interfaces[interfaces.count - 1].endpoints.append((field(2), field(3)))
        default:
          break
        }
        index += length
      }
      let usable = interfaces.filter { $0.alternate == 0 }
      guard let chosen = usable.first(where: { $0.kind == [0xff, 0xff, 0xff] }) ?? usable.first(where: { $0.number == 0 }) else {
        throw LinkError.closed("the gadget's descriptors have no vendor interface")
      }
      let bulk = chosen.endpoints.filter { $0.attributes & 0x03 == 0x02 }
      guard let input = bulk.first(where: { $0.address & 0x80 != 0 }), let output = bulk.first(where: { $0.address & 0x80 == 0 }) else {
        throw LinkError.closed("interface \(chosen.number) of the gadget has no bulk pair")
      }
      return GadgetDescriptors(interface: chosen.number, inEndpoint: input.address, outEndpoint: output.address)
    }
  }

  /// usbdevfs on /dev/bus/usb.
  struct DevUsbfs: UsbfsNode {
    func open(_ path: String) throws(KernelError) -> Int32 {
      let fd = Glibc.open(path, O_RDWR | O_CLOEXEC)
      guard fd >= 0 else { throw KernelError(what: path, errno: errno) }
      return fd
    }

    func descriptors(_ fd: Int32) throws(KernelError) -> [UInt8] {
      var bytes = [UInt8](repeating: 0, count: 8192)
      let n = bytes.withUnsafeMutableBytes { jl_usbfs_descriptors(fd, $0.baseAddress, Int32($0.count)) }
      guard n >= 0 else { throw KernelError(what: "the gadget's descriptors", errno: -n) }
      return Array(bytes[..<Int(n)])
    }

    func driver(_ fd: Int32, interface: UInt8) throws(KernelError) -> String? {
      var name = [CChar](repeating: 0, count: 256)
      let error = name.withUnsafeMutableBufferPointer { jl_usbfs_driver(fd, UInt32(interface), $0.baseAddress, Int32($0.count)) }
      if error == ENODATA { return nil }
      guard error == 0 else { throw KernelError(what: "the driver of interface \(interface)", errno: error) }
      return name.withUnsafeBufferPointer { String(cString: $0.baseAddress!) }
    }

    func disconnect(_ fd: Int32, interface: UInt8) throws(KernelError) {
      let error = jl_usbfs_disconnect(fd, UInt32(interface))
      guard error == 0 else { throw KernelError(what: "detaching the driver of interface \(interface)", errno: error) }
    }

    func claim(_ fd: Int32, interface: UInt8) throws(KernelError) {
      let error = jl_usbfs_claim(fd, UInt32(interface))
      guard error == 0 else { throw KernelError(what: "claiming interface \(interface)", errno: error) }
    }

    func release(_ fd: Int32, interface: UInt8) {
      // ENODEV once the gadget is gone, and the close lets go of it anyway.
      _ = jl_usbfs_release(fd, UInt32(interface))
    }

    func close(_ fd: Int32) {
      _ = Glibc.close(fd)
    }
  }
#endif
