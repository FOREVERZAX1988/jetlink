#if os(Linux)
  import Foundation

  /// One of the kernel's thermal zones: its type ("tj-thermal") and where
  /// its temperature is.
  public struct ThermalZone: Sendable, Equatable {
    public let zone: String
    public let type: String
    public let tempPath: String

    /// Degrees C, or nil when the zone has no reading: an Orin's cv* zones
    /// have no sensor wired up and answer ENODATA.
    public func celsius() -> Double? {
      Sysfs.readInt(tempPath).map { Double($0) / 1000 }
    }

    /// Every zone, in `sorted(glob)` order.
    public static func all(_ root: HostRoot = .system) -> [ThermalZone] {
      let base = "/sys/devices/virtual/thermal"
      return root.list(base).filter { $0.hasPrefix("thermal_zone") }.compactMap { zone in
        guard let type = Sysfs.read(root.path("\(base)/\(zone)/type")), !type.isEmpty else { return nil }
        return ThermalZone(zone: zone, type: type, tempPath: root.path("\(base)/\(zone)/temp"))
      }
    }
  }

  /// A hwmon device's directory by its driver's name, the first in
  /// `sorted(glob)` order: ina3221 (the power monitor), pwm_tach (the fan).
  public func hwmon(named name: String, _ root: HostRoot = .system) -> String? {
    root.list("/sys/class/hwmon").lazy.map { root.path("/sys/class/hwmon/\($0)") }.first { Sysfs.read("\($0)/name") == name }
  }

  /// One INA3221 channel: a rail's label, millivolts and milliamps.
  public struct PowerRail: Sendable, Equatable {
    public let channel: Int
    public let label: String
    public let millivolts: Int
    public let milliamps: Int

    /// The channels with a label, from an ina3221 hwmon directory. Channel 1
    /// is VDD_IN on an Orin: the whole board's input.
    public static func all(ina3221 directory: String) -> [PowerRail] {
      (1...3).compactMap { channel in
        guard let label = Sysfs.read("\(directory)/in\(channel)_label"), !label.isEmpty else { return nil }
        return PowerRail(
          channel: channel, label: label, millivolts: Sysfs.readInt("\(directory)/in\(channel)_input") ?? 0,
          milliamps: Sysfs.readInt("\(directory)/curr\(channel)_input") ?? 0)
      }
    }
  }

  /// A Jetson's health from Tegra sysfs, in the keys the comma logs as
  /// `jetlinkTelemetry` (Python's `telemetry.Telemetry`). Paths are found
  /// once; a sample is then a handful of small reads, each of which may fail
  /// and reads 0.
  public struct TegraTelemetry: Sendable {
    /// What the comma's UI measures power against: the Orin Nano Super's
    /// 25 W mode.
    public static let powerLimitW = 25.0

    let zones: [String: String]
    let ina3221: String?
    let fan: String?
    let gpuLoad: String
    let gpuFrequency: String

    public init(root: HostRoot = .system) {
      var zones: [String: String] = [:]
      for zone in ThermalZone.all(root) {
        zones[zone.type] = zone.tempPath
      }
      self.zones = zones
      ina3221 = hwmon(named: "ina3221", root)
      fan = hwmon(named: "pwm_tach", root)
      gpuLoad = root.path("/sys/devices/platform/bus@0/17000000.gpu/load")
      gpuFrequency = root.path("/sys/class/devfreq/17000000.gpu/cur_freq")
    }

    public func read() -> [String: Any] {
      // tj-thermal is the junction temperature the throttle point is defined
      // on, the nearest thing to chestnut's GPU hotspot. As in Python, a zone
      // reading 0 counts as none.
      let temp = celsius(["tj-thermal", "gpu-thermal"])
      let memoryTemp = celsius(["soc0-thermal", "cpu-thermal"])
      let mv = ina3221.flatMap { Sysfs.readInt("\($0)/in1_input") } ?? 0
      let ma = ina3221.flatMap { Sysfs.readInt("\($0)/curr1_input") } ?? 0
      let loadPermille = Sysfs.readInt(gpuLoad) ?? 0
      let frequency = Sysfs.readInt(gpuFrequency) ?? 0
      return [
        "temp_c": rounded(temp, 1),
        "memory_temp_c": rounded(memoryTemp, 1),
        "power_w": rounded(Double(mv * ma) / 1e6, 2),
        "power_limit_w": TegraTelemetry.powerLimitW,
        "gpu_load_pct": min(100, loadPermille / 10),
        "gpu_clock_mhz": frequency / 1_000_000,
        "fan_rpm": fan.flatMap { Sysfs.readInt("\($0)/rpm") } ?? 0,
        "supply_mv": mv,
        "supply_ma": ma,
      ]
    }

    private func celsius(_ types: [String]) -> Double {
      for type in types {
        if let path = zones[type], let milli = Sysfs.readInt(path), milli != 0 {
          return Double(milli) / 1000
        }
      }
      return 0
    }
  }

  /// Python's round(x, digits) on the values it sees: ties to even.
  func rounded(_ value: Double, _ digits: Int) -> Double {
    let scale = pow(10, Double(digits))
    return (value * scale).rounded(.toNearestOrEven) / scale
  }
#endif
