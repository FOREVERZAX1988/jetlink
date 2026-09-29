import Foundation

/// A request as far as the page reads one: its request line. Headers are
/// read past, never used, and with the line capped in size.
enum PageRequest: Equatable {
  /// No blank line yet: read more.
  case incomplete
  /// The request line and headers passed the cap before their blank line.
  case tooLarge
  case bad
  /// The path, without its query.
  case request(method: String, path: String)

  /// The request in `bytes`, whose request line and headers may take up to
  /// `limit` bytes with the blank line that ends them.
  static func parse(_ bytes: [UInt8], limit: Int) -> PageRequest {
    let window = bytes.prefix(limit)
    guard let end = headerEnd(window) else {
      return bytes.count >= limit ? .tooLarge : .incomplete
    }
    let head = window[..<end]
    let lineEnd = head.firstIndex(of: 0x0A) ?? head.endIndex
    var line = head[..<lineEnd]
    if line.last == 0x0D { line = line.dropLast() }
    guard line.allSatisfy({ $0 >= 0x20 && $0 < 0x7F }) else { return .bad }
    let parts = String(decoding: line, as: UTF8.self).split(separator: " ", omittingEmptySubsequences: false)
    guard parts.count == 3, parts[2].hasPrefix("HTTP/1."), !parts[0].isEmpty, parts[1].hasPrefix("/") else { return .bad }
    let target = parts[1]
    let path = target[..<(target.firstIndex { $0 == "?" || $0 == "#" } ?? target.endIndex)]
    return .request(method: String(parts[0]), path: String(path))
  }

  /// Where the blank line after the headers starts, CRLF or bare LF.
  private static func headerEnd(_ bytes: ArraySlice<UInt8>) -> Int? {
    var index = bytes.startIndex
    while index < bytes.endIndex {
      if bytes[index] == 0x0A {
        let next = bytes.index(after: index)
        if next < bytes.endIndex && bytes[next] == 0x0A { return index }
        if next + 1 < bytes.endIndex && bytes[next] == 0x0D && bytes[next + 1] == 0x0A { return index }
      }
      index += 1
    }
    return nil
  }
}

/// A whole response: status, a few headers, the body. Every connection
/// closes after its one response; an event stream is one long response.
enum PageResponse {
  static let reasons = [
    200: "OK", 400: "Bad Request", 404: "Not Found", 408: "Request Timeout", 431: "Request Header Fields Too Large",
    503: "Service Unavailable",
  ]

  static func head(_ status: Int, type: String, length: Int? = nil) -> Data {
    var text = "HTTP/1.1 \(status) \(reasons[status] ?? "Error")\r\nContent-Type: \(type)\r\n"
    if let length { text += "Content-Length: \(length)\r\n" }
    text += "Cache-Control: no-cache\r\nConnection: close\r\n\r\n"
    return Data(text.utf8)
  }

  static func whole(_ status: Int, type: String, body: Data) -> Data {
    head(status, type: type, length: body.count) + body
  }

  static func text(_ status: Int, _ body: String) -> Data {
    whole(status, type: "text/plain; charset=utf-8", body: Data(body.utf8))
  }
}
