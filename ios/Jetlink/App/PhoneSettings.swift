import Foundation
import JetlinkServer
import Observation

/// The few things worth changing on a phone, kept in UserDefaults.
@MainActor
@Observable
final class PhoneSettings {
  private let defaults: UserDefaults

  /// The TCP port the comma's JetlinkEndpoint names.
  var port: UInt16 {
    didSet { defaults.set(Int(port), forKey: Keys.port) }
  }

  /// The vision trunk on the Neural Engine and the rest on the GPU, or the
  /// whole model on the GPU for when something else holds the Neural Engine.
  var device: CoreMLBackend.Device {
    didSet { defaults.set(device.rawValue, forKey: Keys.device) }
  }

  /// A small GPU job between frames so the GPU does not clock down in the gaps.
  var keepGPUAwake: Bool {
    didSet { defaults.set(keepGPUAwake, forKey: Keys.keepGPUAwake) }
  }

  /// The screen stays on while Jetlink is open: iOS suspends an app whose
  /// phone locks, and a suspended app serves nothing.
  var keepScreenOn: Bool {
    didSet { defaults.set(keepScreenOn, forKey: Keys.keepScreenOn) }
  }

  init(defaults: UserDefaults = .standard) {
    self.defaults = defaults
    let port = defaults.integer(forKey: Keys.port)
    self.port = port > 0 && port < 65_536 ? UInt16(port) : 5599
    self.device = defaults.string(forKey: Keys.device).flatMap(CoreMLBackend.Device.init(rawValue:)) ?? .ane
    self.keepGPUAwake = defaults.object(forKey: Keys.keepGPUAwake) as? Bool ?? true
    self.keepScreenOn = defaults.object(forKey: Keys.keepScreenOn) as? Bool ?? true
  }

  /// Where models and prepared engines live. Application Support, not Caches:
  /// iOS empties Caches when space runs low, and a missing engine costs the
  /// comma its big model on the next drive.
  static var cacheDirectory: URL {
    URL.applicationSupportDirectory.appending(path: "Jetlink/cache", directoryHint: .isDirectory)
  }

  private enum Keys {
    static let port = "port"
    static let device = "device"
    static let keepGPUAwake = "keepGPUAwake"
    static let keepScreenOn = "keepScreenOn"
  }
}

extension CoreMLBackend.Device {
  var title: String {
    switch self {
    case .ane: "Neural Engine and GPU"
    case .coreml: "GPU only"
    case .cpu: "CPU only"
    }
  }
}
