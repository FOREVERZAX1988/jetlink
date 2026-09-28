import Foundation
import Testing

@testable import JetlinkStatusPage

@Suite("Status page requests")
struct PageHTTPTests {
  func parse(_ text: String, limit: Int = 8192) -> PageRequest {
    PageRequest.parse(Array(text.utf8), limit: limit)
  }

  @Test("A request line, its headers and the blank line")
  func request() {
    #expect(parse("GET /events HTTP/1.1\r\nHost: jetlink.local\r\nAccept: text/event-stream\r\n\r\n") == .request(method: "GET", path: "/events"))
    #expect(parse("GET /?reload=1 HTTP/1.0\n\n") == .request(method: "GET", path: "/"))
    #expect(parse("POST /logs HTTP/1.1\r\n\r\n") == .request(method: "POST", path: "/logs"))
  }

  @Test("Nothing is decided before the blank line")
  func incomplete() {
    #expect(parse("") == .incomplete)
    #expect(parse("GET / HTTP/1.1\r\n") == .incomplete)
    #expect(parse("GET / HTTP/1.1\r\nHost: x\r\n") == .incomplete)
  }

  @Test("The request line and headers are capped, blank line included")
  func capped() {
    let head = "GET / HTTP/1.1\r\nX-Pad: "
    let fits = head + String(repeating: "a", count: 8192 - head.utf8.count - 4) + "\r\n\r\n"
    #expect(fits.utf8.count == 8192)
    #expect(parse(fits) == .request(method: "GET", path: "/"))
    let over = head + String(repeating: "a", count: 8192 - head.utf8.count - 3) + "\r\n\r\n"
    #expect(parse(over) == .tooLarge)
    #expect(parse(String(repeating: "a", count: 9000)) == .tooLarge)
  }

  @Test("A malformed request line is refused")
  func bad() {
    #expect(parse("HELLO\r\n\r\n") == .bad)
    #expect(parse("GET /\r\n\r\n") == .bad)
    #expect(parse("GET http://jetlink.local/ HTTP/1.1\r\n\r\n") == .bad)
    #expect(parse("GET / SPDY/3\r\n\r\n") == .bad)
    #expect(PageRequest.parse(Array("GET /\u{1} HTTP/1.1\r\n\r\n".utf8), limit: 8192) == .bad)
  }

  @Test("A reply closes its connection and says how long it is")
  func reply() {
    let text = String(decoding: PageResponse.text(404, "not found\n"), as: UTF8.self)
    #expect(text.hasPrefix("HTTP/1.1 404 Not Found\r\n"))
    #expect(text.contains("Content-Length: 10\r\n"))
    #expect(text.contains("Connection: close\r\n"))
    #expect(text.hasSuffix("\r\n\r\nnot found\n"))
  }
}
