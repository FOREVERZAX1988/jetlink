import Foundation
import Observation
import UIKit

/// Keeps `DeviceHealth` current: the thermal state, the battery, Low Power Mode.
@MainActor
@Observable
final class DeviceMonitor {
  private(set) var health = DeviceHealth()
  @ObservationIgnored private var observers: [NSObjectProtocol] = []

  init() {
    UIDevice.current.isBatteryMonitoringEnabled = true
    refresh()
    let center = NotificationCenter.default
    for name in [
      ProcessInfo.thermalStateDidChangeNotification,
      Notification.Name.NSProcessInfoPowerStateDidChange,
      UIDevice.batteryLevelDidChangeNotification,
      UIDevice.batteryStateDidChangeNotification,
    ] {
      observers.append(
        center.addObserver(forName: name, object: nil, queue: .main) { [weak self] _ in
          MainActor.assumeIsolated { self?.refresh() }
        })
    }
  }

  func refresh() {
    let device = UIDevice.current
    let power: DeviceHealth.Power =
      switch device.batteryState {
      case .charging: .charging
      case .full: .full
      case .unplugged: .unplugged
      case .unknown: .unknown
      @unknown default: .unknown
      }
    health = DeviceHealth(
      thermal: DeviceHealth.Thermal(ProcessInfo.processInfo.thermalState),
      batteryLevel: device.batteryLevel >= 0 ? Double(device.batteryLevel) : nil,
      power: power,
      lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled)
  }
}
