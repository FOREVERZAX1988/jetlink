import Foundation
import SwiftUI

/// What the phone itself is doing, for the tiles under the frame budget. A
/// phone in a car mount in the sun throttles, and a throttled phone misses
/// frames long before the numbers above say why.
struct DeviceHealth: Equatable, Sendable {
  enum Thermal: Equatable, Sendable {
    case nominal, fair, serious, critical

    init(_ state: ProcessInfo.ThermalState) {
      switch state {
      case .nominal: self = .nominal
      case .fair: self = .fair
      case .serious: self = .serious
      case .critical: self = .critical
      @unknown default: self = .fair
      }
    }

    var title: String {
      switch self {
      case .nominal: "Cool"
      case .fair: "Warm"
      case .serious: "Hot"
      case .critical: "Too hot"
      }
    }

    /// What the state means for the frames, in a few words.
    var detail: String {
      switch self {
      case .nominal: "Full speed"
      case .fair: "Full speed"
      case .serious: "Slowing down"
      case .critical: "Throttled hard"
      }
    }

    var symbol: String {
      switch self {
      case .nominal: "thermometer.low"
      case .fair: "thermometer.medium"
      case .serious: "thermometer.high"
      case .critical: "flame.fill"
      }
    }

    var tone: Color {
      switch self {
      case .nominal, .fair: .green
      case .serious: .orange
      case .critical: .red
      }
    }
  }

  enum Power: Equatable, Sendable {
    case unknown, unplugged, charging, full
  }

  var thermal: Thermal = .nominal
  /// 0 to 1, nil where the level is not known.
  var batteryLevel: Double?
  var power: Power = .unknown
  var lowPowerMode = false

  var batteryText: String {
    guard let batteryLevel else { return "Unknown" }
    return batteryLevel.formatted(.percent.precision(.fractionLength(0)))
  }

  var powerText: String {
    switch power {
    case .unknown: "Power unknown"
    case .unplugged: lowPowerMode ? "Low Power Mode" : "Not charging"
    case .charging: "Charging"
    case .full: "Charged"
    }
  }

  var batterySymbol: String {
    switch power {
    case .charging, .full: return "battery.100percent.bolt"
    case .unknown: return "battery.0percent"
    case .unplugged:
      let level = batteryLevel ?? 0
      if level > 0.75 { return "battery.100percent" }
      if level > 0.5 { return "battery.75percent" }
      if level > 0.25 { return "battery.50percent" }
      if level > 0.1 { return "battery.25percent" }
      return "battery.0percent"
    }
  }

  /// A phone running a model twenty times a second belongs on power.
  var batteryTone: Color {
    switch power {
    case .charging, .full: .green
    case .unknown: .secondary
    case .unplugged: (batteryLevel ?? 1) < 0.2 ? .red : .orange
    }
  }
}
