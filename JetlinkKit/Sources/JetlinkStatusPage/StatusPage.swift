import Foundation

/// The read-only status page the daemon serves to a phone on the comma's
/// hotspot. The page is a resource of this module, in a bundle SwiftPM puts
/// beside the executable (`JetlinkKit_JetlinkStatusPage.resources` on Linux).
public enum StatusPage {
  /// The page itself.
  public static func page() throws -> Data {
    guard let url = Bundle.module.url(forResource: "index", withExtension: "html", subdirectory: "Resources") else {
      throw CocoaError(.fileReadNoSuchFile)
    }
    return try Data(contentsOf: url)
  }
}
