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

  @Test("A binary without the page's bundle is told so rather than stopped")
  func missing() {
    let nowhere = FileManager.default.temporaryDirectory.appendingPathComponent("jetlink-no-page-\(UUID().uuidString)")
    let error = #expect(throws: StatusPage.Unavailable.self) { try StatusPage.page(searching: [nowhere]) }
    #expect(error?.searched == [nowhere.path])
  }
}
