import Foundation
import Synchronization

/// Mirror bases the registry's requests try before the original hosts, and
/// the shapes of URL each base can stand in for. A phone in a network that
/// blocks GitHub and HuggingFace reaches both through these; a phone that
/// reaches the originals never notices, because the originals come last.
///
/// Two shapes of base:
/// - a HuggingFace reverse proxy, such as `https://hf-mirror.com`: the URL's
///   `huggingface.co` host is swapped for the base's, path and query intact.
///   Its `/api`, its `resolve` and its LFS batch endpoint all proxy the same.
/// - a URL prefix, such as `https://ghfast.top`: the whole original URL is
///   appended, which is how the GitHub speed-up proxies take it. Tried only
///   for GitHub's hosts.
///
/// The bases are the app's mirror setting (Settings.kt), handed over the
/// start config; the order is the user's, and the original URL is always the
/// last candidate, so a list without mirrors changes nothing.
public enum Mirrors {
  /// The configured bases, in the user's order. Set once at start, read from
  /// the network tasks; a lock because start can race a download in flight.
  private static let state = Mutex<[String]>([])

  /// Which hosts a prefix mirror is asked to carry.
  private static let githubHosts: Set<String> = [
    "github.com", "raw.githubusercontent.com", "objects.githubusercontent.com",
    "media.githubusercontent.com", "codeload.github.com", "gist.githubusercontent.com",
  ]

  public static func configure(_ bases: [String]) {
    state.withLock { $0 = bases.compactMap { clean($0) } }
  }

  public static func configured() -> [String] {
    state.withLock { $0 }
  }

  /// scheme://host, with anything after the host dropped, or nil.
  static func clean(_ base: String) -> String? {
    guard let schemeEnd = base.range(of: "://") else { return nil }
    let scheme = String(base[..<schemeEnd.lowerBound])
    let rest = base[schemeEnd.upperBound...]
    let host = rest.prefix { $0 != "/" && $0 != "?" && $0 != "#" }
    guard !host.isEmpty else { return nil }
    return scheme + "://" + String(host)
  }

  static func host(of url: String) -> String? {
    guard let schemeEnd = url.range(of: "://") else { return nil }
    return String(url[schemeEnd.upperBound...].prefix { $0 != "/" && $0 != "?" && $0 != "#" }).lowercased()
  }

  /// The shape of a mirror base, or nil when it cannot carry anything.
  static func shape(of base: String) -> Shape? {
    guard let baseHost = host(of: base) else { return nil }
    if baseHost == "hf-mirror.com" || baseHost.hasSuffix(".hf-mirror.com") {
      return .huggingFaceProxy
    }
    return .githubPrefix
  }

  enum Shape {
    case huggingFaceProxy
    case githubPrefix
  }

  /// The rewritten URL a base serves this one at, or nil when the base
  /// cannot carry that host.
  static func rewrite(_ base: String, for url: String) -> String? {
    switch shape(of: base) {
    case .huggingFaceProxy:
      guard host(of: url) == "huggingface.co" else { return nil }
      return url.replacingOccurrences(of: "https://huggingface.co/", with: "\(base)/")
    case .githubPrefix:
      guard let urlHost = host(of: url), githubHosts.contains(urlHost) else { return nil }
      return "\(base)/\(url)"
    case nil:
      return nil
    }
  }

  /// The URLs one request tries, in order: every mirror that can carry it,
  /// then the original itself. Without mirrors, just the original.
  public static func candidates(for url: String) -> [String] {
    var found: [String] = []
    for base in configured() {
      if let rewritten = rewrite(base, for: url), rewritten != url, !found.contains(rewritten) {
        found.append(rewritten)
      }
    }
    found.append(url)
    return found
  }
}
