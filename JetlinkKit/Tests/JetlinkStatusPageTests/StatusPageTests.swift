import Foundation
import Testing

@testable import JetlinkStatusPage

@Suite("Status page")
struct StatusPageTests {
  @Test("The page is bundled with the module")
  func bundled() throws {
    let page = String(decoding: try StatusPage.page(), as: UTF8.self)
    #expect(page.hasPrefix("<!doctype html>"))
  }

  @Test("The page needs nothing but this server: the comma's hotspot has no internet")
  func offline() throws {
    let page = String(decoding: try StatusPage.page(), as: UTF8.self)
    #expect(!page.contains("http://") && !page.contains("https://") && !page.contains("//cdn"))
    #expect(page.range(of: #"(src|href)="(?!data:)"#, options: .regularExpression) == nil)
    #expect(page.contains("new EventSource('events')") && page.contains("fetch('logs'"))
    // It reads every event the feed relays, and the host's two.
    for name in ["host", "hello", "server", "link", "engine", "inventory", "stats", "hw"] {
      #expect(page.contains("case '\(name)':"), "\(name)")
    }
    #expect(!page.contains("\u{2014}"))
  }

  @Test("A binary without the page's bundle is told so rather than stopped")
  func missing() {
    let nowhere = FileManager.default.temporaryDirectory.appendingPathComponent("jetlink-no-page-\(UUID().uuidString)")
    let error = #expect(throws: StatusPage.Unavailable.self) { try StatusPage.page(searching: [nowhere]) }
    #expect(error?.searched == [nowhere.path])
  }
}
