import Foundation
import JetlinkServer
import SwiftUI
import Testing

@testable import Jetlink

/// The Swift server the app can run in place of the Python one: which server a
/// stored setting means, and what the Swift server is asked to be.
struct EmbeddedServerTests {
  private let cache = URL(filePath: "/Users/me/Library/Application Support/Jetlink/cache")

  @MainActor @Test func thePythonServerStaysTheDefault() throws {
    let suite = "io.zoompilot.jetlink.tests.\(UUID().uuidString)"
    let defaults = try #require(UserDefaults(suiteName: suite))
    defer { defaults.removePersistentDomain(forName: suite) }
    #expect(AppSettings(defaults: defaults).serverEngine == .python)
    defaults.set("swift", forKey: AppSettings.Key.serverEngine)
    #expect(AppSettings(defaults: defaults).serverEngine == .swift)
    defaults.set("rust", forKey: AppSettings.Key.serverEngine)
    #expect(AppSettings(defaults: defaults).serverEngine == .python)
  }

  @Test func tinygradIsPythonOnly() {
    #expect(ServerEngine.python.backends.contains(.tinygrad))
    #expect(!ServerEngine.swift.backends.contains(.tinygrad))
  }

  @Test func usbServesTheGadgetAndOpensNoPort() {
    let configuration = ServerStore.embeddedConfiguration(backend: .auto, transport: .usb, tcpPort: 5599, cacheDirectory: cache)
    #expect(configuration.usb)
    #expect(!configuration.listen)
    #expect(configuration.cacheRoot == cache)
  }

  @Test func tcpListensOnThePortAndLeavesUSBAlone() {
    let configuration = ServerStore.embeddedConfiguration(backend: .auto, transport: .tcp, tcpPort: 5601, cacheDirectory: cache)
    #expect(!configuration.usb)
    #expect(configuration.listen)
    #expect(configuration.port == 5601)
  }

  @Test(arguments: [
    (BackendChoice.auto, CoreMLBackend.Device.ane),
    (BackendChoice.coreml, CoreMLBackend.Device.coreml),
    (BackendChoice.tinygrad, CoreMLBackend.Device.ane),
  ])
  func backendMapping(choice: BackendChoice, device: CoreMLBackend.Device) {
    #expect(ServerStore.embeddedConfiguration(backend: choice, transport: .usb, tcpPort: 5599, cacheDirectory: cache).device == device)
  }

  @Test func logLinesReadLikePythons() throws {
    var parts = DateComponents()
    (parts.year, parts.month, parts.day, parts.hour, parts.minute, parts.second, parts.nanosecond) = (2026, 9, 27, 13, 4, 5, 123_000_000)
    let date = try #require(Calendar.current.date(from: parts))
    #expect(EmbeddedServer.logLine(.warning, "server", "hello", at: date) == "2026-09-27 13:04:05,123 WARNING jetlink.server: hello")
    #expect(EmbeddedServer.logLine(.info, "usb", "x", at: date) == "2026-09-27 13:04:05,123 INFO    jetlink.usb: x")
    #expect(LogsView.tone(for: EmbeddedServer.logLine(.error, "session", "bad", at: date)) == .red)
  }
}
