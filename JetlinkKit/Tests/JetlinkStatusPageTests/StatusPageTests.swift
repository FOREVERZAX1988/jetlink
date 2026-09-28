import Foundation
import JetlinkStatusPage
import Testing

@Suite("Status page")
struct StatusPageTests {
  @Test("The page is bundled with the module")
  func bundled() throws {
    let page = String(decoding: try StatusPage.page(), as: UTF8.self)
    #expect(page.hasPrefix("<!doctype html>"))
  }
}
