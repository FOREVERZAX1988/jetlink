#if os(Linux)
  import Foundation
  import Glibc
  import JetlinkKit
  import Testing

  @testable import JetlinkLinux
  @testable import JetlinkServer

  /// Descriptors as a usbfs node reads them back.
  enum Descriptors {
    static let device: [UInt8] = [18, 1, 0x20, 0x03, 0, 0, 0, 9, 0x09, 0x12, 0x01, 0x00, 0x00, 0x00, 1, 2, 0, 1]

    static func endpoint(_ address: UInt8, bulk: Bool = true) -> [UInt8] {
      // wMaxPacketSize 1024, and the SuperSpeed companion a 5 Gb/s link adds.
      [7, 5, address, bulk ? 0x02 : 0x03, 0x00, 0x04, 0] + [6, 0x30, 0, 0, 0, 0]
    }

    static func interface(_ number: UInt8, alternate: UInt8 = 0, kind: [UInt8] = [0xff, 0xff, 0xff], _ endpoints: [UInt8]...) -> [UInt8] {
      [9, 4, number, alternate, UInt8(endpoints.count)] + kind + [0] + endpoints.flatMap { $0 }
    }

    static func configuration(_ value: UInt8, _ interfaces: [UInt8]...) -> [UInt8] {
      let body = interfaces.flatMap { $0 }
      let total = 9 + body.count
      let count = UInt8(interfaces.count)
      return [9, 2, UInt8(total & 0xff), UInt8(total >> 8), count, value, 0, 0x80, 50] + body
    }

    /// The comma's jetlink gadget as the Jetson saw it: one FF/FF/FF
    /// interface with bulk 81 in and 01 out.
    static let comma = device + configuration(1, interface(0, endpoint(0x81), endpoint(0x01)))
  }

  /// usbdevfs where the node is a file holding descriptors: opens and reads
  /// are real, the ioctls are the test's.
  final class FakeNode: UsbfsNode, @unchecked Sendable {
    let calls = Dial<[String]>([])
    let bound = Dial<String?>(nil)
    let claimError = Dial<Int32>(0)
    let descriptorsOpen = Dial<Set<Int32>>([])

    func open(_ path: String) throws(KernelError) -> Int32 {
      let fd = try DevUsbfs().open(path)
      calls.value.append("open \(URL(fileURLWithPath: path).pathComponents.suffix(2).joined(separator: "/"))")
      descriptorsOpen.value.insert(fd)
      return fd
    }

    func descriptors(_ fd: Int32) throws(KernelError) -> [UInt8] {
      try DevUsbfs().descriptors(fd)
    }

    func driver(_ fd: Int32, interface: UInt8) throws(KernelError) -> String? {
      bound.value
    }

    func disconnect(_ fd: Int32, interface: UInt8) throws(KernelError) {
      calls.value.append("disconnect \(interface)")
      bound.value = nil
    }

    func claim(_ fd: Int32, interface: UInt8) throws(KernelError) {
      if claimError.value != 0 { throw KernelError(what: "claiming interface \(interface)", errno: claimError.value) }
      calls.value.append("claim \(interface)")
    }

    func release(_ fd: Int32, interface: UInt8) {
      calls.value.append("release \(interface)")
    }

    func close(_ fd: Int32) {
      calls.value.append("close")
      descriptorsOpen.value.remove(fd)
      DevUsbfs().close(fd)
    }
  }

  /// The server's end of the claim: what it was handed, and a device that
  /// goes when told, as ENODEV from a reap makes `UsbfsGadget`'s go.
  final class FakeTarget: UsbfsTarget, @unchecked Sendable {
    let attached = Dial<(fd: Int32, input: UInt8, output: UInt8)?>(nil)
    let events = Dial<[String]>([])
    let gone = Dial(false)
    let attachError = Dial<Bool>(false)

    func attach(fd: Int32, inEndpoint: UInt8, outEndpoint: UInt8) throws {
      if attachError.value { throw LinkError.closed("could not create the USB wake event") }
      attached.value = (fd, inEndpoint, outEndpoint)
      gone.value = false
      events.value.append(String(format: "attach %02x %02x", inEndpoint, outEndpoint))
    }

    func detach() {
      attached.value = nil
      events.value.append("detach")
    }

    func present() -> Bool {
      attached.value != nil && !gone.value
    }

    func open() throws -> any MessageLink {
      guard present() else { throw LinkError.closed("no jetlink gadget is attached") }
      return StubLink()
    }
  }

  final class StubLink: MessageLink {
    var peer: String { "usb" }
    var medium: LinkMedium? { nil }
    var connectsOnOpen: Bool { false }
    func recv() throws -> Message { throw LinkError.closed("stub") }
    func sendParts(_ type: Wire.Msg, seq: UInt32, parts: UnsafeBufferPointer<UnsafeRawBufferPointer>, flags: Wire.Flag) throws {}
    func shutdown() {}
    func close() {}
  }

  /// A USB bus in a temporary tree: sysfs entries and usbfs nodes.
  final class Bus {
    let tree = Tree()
    let node = FakeNode()
    let target = FakeTarget()
    let lines = Lines()
    lazy var gadget = SysfsGadget(root: tree.root, node: node, target: target, log: lines.log)

    init() {
      // A hub and its interface, which the scan passes over.
      device("2-1", vendor: "0bda", product: "0489", bus: 2, device: 2)
      tree.write("/sys/bus/usb/devices/2-1:1.0/bInterfaceClass", "09\n")
    }

    func device(_ name: String, vendor: String, product: String, bus: Int, device: Int, descriptors: [UInt8] = Descriptors.comma) {
      let base = "/sys/bus/usb/devices/\(name)"
      tree.write("\(base)/idVendor", "\(vendor)\n")
      tree.write("\(base)/idProduct", "\(product)\n")
      tree.write("\(base)/busnum", "\(bus)\n")
      tree.write("\(base)/devnum", "\(device)\n")
      tree.write("\(base)/speed", "5000\n")
      tree.write(String(format: "/dev/bus/usb/%03d/%03d", bus, device), bytes: descriptors)
    }

    /// The comma's gadget plugs in behind the hub.
    func plug(device number: Int = 3, descriptors: [UInt8] = Descriptors.comma) {
      device("2-1.3", vendor: "1209", product: "0001", bus: 2, device: number, descriptors: descriptors)
    }

    func unplug(device number: Int = 3) {
      tree.remove("/sys/bus/usb/devices/2-1.3")
      tree.remove(String(format: "/dev/bus/usb/002/%03d", number))
    }
  }

  @Suite("Sysfs gadget")
  struct SysfsGadgetTests {
    @Test("The journal line install.sh and jetlink status wait for is unchanged")
    func waitingLine() {
      #expect(Server.usbWaiting == "waiting for a jetlink gadget at 1209:0001")
    }

    @Test("Found in the bench Jetson's sysfs: 2-1.3, bus 2 device 3")
    func capture() {
      let gadget = SysfsGadget(root: jetson, node: FakeNode(), target: FakeTarget(), log: Lines().log)
      #expect(gadget.scan() == SysfsGadget.Found(name: "2-1.3", bus: 2, device: 3, speed: "5000"))
      #expect(gadget.nodePath(gadget.scan()!).hasSuffix("/dev/bus/usb/002/003"))
    }

    @Test("Nothing on the bus: not present, and nothing opened")
    func absent() {
      let bus = Bus()
      #expect(!bus.gadget.present())
      #expect(throws: LinkError.self) { try bus.gadget.open() }
      #expect(bus.node.calls.value.isEmpty)
    }

    @Test("First sight: opens the node sysfs names, claims the vendor interface, attaches its bulk pair")
    func claims() throws {
      let bus = Bus()
      bus.plug()
      #expect(bus.gadget.present())
      // Presence alone opens nothing: a comma that cannot be claimed is still there.
      #expect(bus.node.calls.value.isEmpty)
      _ = try bus.gadget.open()
      #expect(bus.node.calls.value == ["open 002/003", "claim 0"])
      #expect(bus.target.events.value == ["attach 81 01"])
      #expect(bus.lines.has(.info, "claimed the comma's gadget at 2-1.3"))
    }

    @Test("The descriptor stays open across the comma's per-run reopens")
    func keepsTheDescriptor() throws {
      let bus = Bus()
      bus.plug()
      for _ in 0..<5 {
        #expect(bus.gadget.present())
        _ = try bus.gadget.open()
      }
      #expect(bus.node.calls.value == ["open 002/003", "claim 0"])
      #expect(bus.target.events.value == ["attach 81 01"])
    }

    @Test("Gone from sysfs: detached, released, closed")
    func unplugged() throws {
      let bus = Bus()
      bus.plug()
      #expect(bus.gadget.present())
      _ = try bus.gadget.open()
      bus.unplug()
      #expect(!bus.gadget.present())
      #expect(bus.node.calls.value == ["open 002/003", "claim 0", "release 0", "close"])
      #expect(bus.target.events.value == ["attach 81 01", "detach"])
      #expect(bus.node.descriptorsOpen.value.isEmpty)
      #expect(bus.lines.has(.info, "released the comma's gadget at 2-1.3: it left the bus"))
    }

    @Test("Re-enumerated between polls: the old claim goes, the new device is claimed")
    func reenumerated() throws {
      let bus = Bus()
      bus.plug(device: 3)
      #expect(bus.gadget.present())
      _ = try bus.gadget.open()
      bus.unplug(device: 3)
      bus.plug(
        device: 7,
        descriptors: Descriptors.device + Descriptors.configuration(1, Descriptors.interface(0, Descriptors.endpoint(0x82), Descriptors.endpoint(0x03))))
      #expect(bus.gadget.present())
      _ = try bus.gadget.open()
      #expect(bus.node.calls.value == ["open 002/003", "claim 0", "release 0", "close", "open 002/007", "claim 0"])
      // The endpoints are read afresh: FunctionFS numbers them at every bind.
      #expect(bus.target.events.value == ["attach 81 01", "detach", "attach 82 03"])
    }

    @Test("ENODEV under the transfers with sysfs not caught up yet: released, and claimed again when it opens")
    func goneUnderTransfers() throws {
      let bus = Bus()
      bus.plug()
      #expect(bus.gadget.present())
      _ = try bus.gadget.open()
      bus.target.gone.value = true
      #expect(bus.gadget.present())
      #expect(bus.target.events.value == ["attach 81 01", "detach"])
      _ = try bus.gadget.open()
      #expect(bus.node.calls.value == ["open 002/003", "claim 0", "release 0", "close", "open 002/003", "claim 0"])
    }

    @Test("A claim refused is thrown for the USB loop to log, the node closed, and tried again")
    func busy() throws {
      let bus = Bus()
      bus.plug()
      bus.node.claimError.value = EBUSY
      #expect(bus.gadget.present())
      #expect(throws: KernelError.self) { try bus.gadget.open() }
      #expect(bus.node.calls.value == ["open 002/003", "close"])
      #expect(bus.node.descriptorsOpen.value.isEmpty)
      #expect(bus.target.events.value.isEmpty)
      bus.node.claimError.value = 0
      _ = try bus.gadget.open()
      #expect(bus.target.events.value == ["attach 81 01"])
    }

    @Test("A node that cannot be opened is an error with the path and the errno")
    func unopenable() {
      let bus = Bus()
      bus.plug()
      bus.tree.remove("/dev/bus/usb/002/003")
      #expect(bus.gadget.present())
      let error = #expect(throws: KernelError.self) { try bus.gadget.open() }
      #expect(error?.description.hasSuffix("/dev/bus/usb/002/003: No such file or directory") == true)
    }

    @Test("A kernel driver on the interface is detached first; another program's usbfs claim is not taken")
    func drivers() throws {
      let bus = Bus()
      bus.plug()
      bus.node.bound.value = "cdc_ncm"
      #expect(bus.gadget.present())
      _ = try bus.gadget.open()
      #expect(bus.node.calls.value == ["open 002/003", "disconnect 0", "claim 0"])

      let other = Bus()
      other.plug()
      other.node.bound.value = "usbfs"
      #expect(other.gadget.present())
      #expect(throws: LinkError.self) { try other.gadget.open() }
      #expect(other.node.calls.value == ["open 002/003", "close"])
    }

    @Test("An attach that fails lets go of the interface and the node")
    func attachFails() {
      let bus = Bus()
      bus.plug()
      bus.target.attachError.value = true
      #expect(bus.gadget.present())
      #expect(throws: LinkError.self) { try bus.gadget.open() }
      #expect(bus.node.calls.value == ["open 002/003", "claim 0", "release 0", "close"])
    }

    @Test("With the real UsbfsGadget behind it, the claim serves a USB link")
    func realGadget() throws {
      let bus = Bus()
      bus.plug()
      let usbfs = UsbfsGadget()
      let gadget = SysfsGadget(root: bus.tree.root, node: bus.node, target: usbfs, log: bus.lines.log)
      #expect(!usbfs.present())
      #expect(gadget.present())
      let link = try gadget.open()
      #expect(link is USBTransport)
      #expect(usbfs.present())
      link.close()
      bus.unplug()
      #expect(!gadget.present())
      #expect(!usbfs.present())
      #expect(bus.node.descriptorsOpen.value.isEmpty)
    }
  }

  @Suite("Gadget descriptors")
  struct GadgetDescriptorsTests {
    typealias D = Descriptors

    @Test("The comma's one interface: 81 in, 01 out")
    func comma() throws {
      #expect(try GadgetDescriptors.pick(D.comma) == GadgetDescriptors(interface: 0, inEndpoint: 0x81, outEndpoint: 0x01))
    }

    @Test("A composite gadget: the FF/FF/FF interface, wherever it is")
    func composite() throws {
      let ncm = D.interface(0, kind: [0x02, 0x0d, 0x00], D.endpoint(0x83, bulk: false))
      let data = D.interface(1, kind: [0x0a, 0x00, 0x01], D.endpoint(0x81), D.endpoint(0x01))
      let vendor = D.interface(2, D.endpoint(0x82), D.endpoint(0x02))
      #expect(
        try GadgetDescriptors.pick(D.device + D.configuration(1, ncm, data, vendor)) == GadgetDescriptors(interface: 2, inEndpoint: 0x82, outEndpoint: 0x02))
    }

    @Test("No vendor interface: interface 0; alternate settings and interrupt endpoints are passed over")
    func fallback() throws {
      let zero = D.interface(0, kind: [0x08, 0x06, 0x50], D.endpoint(0x85, bulk: false), D.endpoint(0x84), D.endpoint(0x04))
      let alternate = D.interface(1, alternate: 1, D.endpoint(0x86), D.endpoint(0x06))
      #expect(
        try GadgetDescriptors.pick(D.device + D.configuration(1, zero, alternate)) == GadgetDescriptors(interface: 0, inEndpoint: 0x84, outEndpoint: 0x04))
    }

    @Test("The active configuration, when sysfs names one")
    func configurations() throws {
      let first = D.configuration(1, D.interface(0, D.endpoint(0x81), D.endpoint(0x01)))
      let second = D.configuration(2, D.interface(0, D.endpoint(0x83), D.endpoint(0x03)))
      #expect(try GadgetDescriptors.pick(D.device + first + second).inEndpoint == 0x81)
      #expect(try GadgetDescriptors.pick(D.device + first + second, configuration: 2).inEndpoint == 0x83)
    }

    @Test("Short, truncated, or without a bulk pair: an error, never a crash")
    func malformed() {
      #expect(throws: LinkError.self) { try GadgetDescriptors.pick([]) }
      #expect(throws: LinkError.self) { try GadgetDescriptors.pick(Array(D.comma.prefix(30))) }
      #expect(throws: LinkError.self) { try GadgetDescriptors.pick(D.device + [0, 2, 0, 0]) }
      #expect(throws: LinkError.self) { try GadgetDescriptors.pick(D.device + D.configuration(1, D.interface(0, D.endpoint(0x81)))) }
    }

    @Test("Read back through jl_usbfs_descriptors from a node's bytes")
    func read() throws {
      let tree = Tree()
      tree.write("/node", bytes: D.comma)
      let node = DevUsbfs()
      let fd = try node.open(tree.path("/node"))
      defer { node.close(fd) }
      #expect(try node.descriptors(fd) == D.comma)
      // Again from the start, as a second claim on the same descriptor reads.
      #expect(try node.descriptors(fd) == D.comma)
      // Not a usbfs node: the ioctls say so with their errno.
      #expect(throws: KernelError.self) { try node.claim(fd, interface: 0) }
    }
  }
#endif
