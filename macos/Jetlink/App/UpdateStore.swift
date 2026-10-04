import AppKit
import Observation
@preconcurrency import Sparkle
import os

/// Updates from the GitHub releases, through Sparkle: a check a day, Check for
/// Updates in the menus, and the two settings. Only a release build has them
/// (`isAvailable`), so a `make app` build never offers to replace itself with
/// a release.
///
/// Nothing installs without the user: Sparkle asks, or with automatic updates
/// on, installs when Jetlink quits. A check that finds an update while a comma
/// is connected keeps it to the menu bar until the comma is gone, so a drive
/// never gets an update alert.
@MainActor
@Observable
final class UpdateStore {
  /// A release build with a Team ID signature (Developer ID for a real one),
  /// a feed and a key to check it with.
  private(set) var isAvailable: Bool
  /// False while a check is running.
  private(set) var canCheckForUpdates = false
  /// The version a scheduled check found while a comma was connected. The
  /// menu bar offers it, and it comes up once the comma is gone and Jetlink is
  /// next in front.
  private(set) var heldUpdate: String?

  /// Sparkle's own settings, read live: the update window can change the
  /// second one too. Set only from the settings view, as Sparkle asks; writing
  /// one at launch would pin Info.plist's default into the user's defaults.
  var automaticallyChecks: Bool {
    get {
      access(keyPath: \.automaticallyChecks)
      return controller?.updater.automaticallyChecksForUpdates ?? false
    }
    set {
      withMutation(keyPath: \.automaticallyChecks) { controller?.updater.automaticallyChecksForUpdates = newValue }
    }
  }
  var automaticallyDownloads: Bool {
    get {
      access(keyPath: \.automaticallyDownloads)
      return controller?.updater.automaticallyDownloadsUpdates ?? false
    }
    set {
      withMutation(keyPath: \.automaticallyDownloads) { controller?.updater.automaticallyDownloadsUpdates = newValue }
    }
  }

  let version: String

  @ObservationIgnored private var controller: SPUStandardUpdaterController?
  @ObservationIgnored private let delegate = UserDriverDelegate()
  @ObservationIgnored private let isCommaConnected: @MainActor () -> Bool
  @ObservationIgnored private var canCheckObservation: NSKeyValueObservation?
  @ObservationIgnored private var activeObserver: NSObjectProtocol?
  @ObservationIgnored private let log = Logger(subsystem: "io.zoompilot.jetlink", category: "updates")

  init(isCommaConnected: @escaping @MainActor () -> Bool, bundle: Bundle = .main) {
    self.isCommaConnected = isCommaConnected
    version = bundle.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "0.0.0"
    isAvailable = UpdateStore.isReleaseBuild(
      version: version,
      feedURL: bundle.object(forInfoDictionaryKey: "SUFeedURL") as? String,
      publicKey: bundle.object(forInfoDictionaryKey: "SUPublicEDKey") as? String,
      teamID: UpdateStore.signingTeam()
    )
    guard isAvailable else {
      log.info("updates are off: \(self.version, privacy: .public) is not a signed release build")
      return
    }
    delegate.store = self
    let controller = SPUStandardUpdaterController(startingUpdater: false, updaterDelegate: nil, userDriverDelegate: delegate)
    do {
      try controller.updater.start()
    } catch {
      log.error("the updater did not start: \(error.localizedDescription, privacy: .public)")
      isAvailable = false
      return
    }
    self.controller = controller
    canCheckObservation = controller.updater.observe(\.canCheckForUpdates, options: [.initial, .new]) { [weak self] _, change in
      let value = change.newValue ?? false
      MainActor.assumeIsolated { self?.canCheckForUpdates = value }
    }
    activeObserver = NotificationCenter.default.addObserver(forName: NSApplication.didBecomeActiveNotification, object: nil, queue: .main) {
      [weak self] _ in
      MainActor.assumeIsolated { self?.showHeldUpdateIfClear() }
    }
  }

  /// Checks now, or brings an update already found to the front.
  func checkForUpdates() {
    controller?.checkForUpdates(nil)
  }

  private func showHeldUpdateIfClear() {
    guard heldUpdate != nil, !isCommaConnected() else { return }
    log.info("showing the update held while a comma was connected")
    checkForUpdates()
  }

  // MARK: - Sparkle's user driver

  fileprivate func shouldSparkleShowScheduledUpdate() -> Bool {
    !isCommaConnected()
  }

  fileprivate func willShowUpdate(_ update: SUAppcastItem, handledBySparkle: Bool) {
    guard !handledBySparkle else { return }
    log.info("holding update \(update.displayVersionString, privacy: .public) while a comma is connected")
    heldUpdate = update.displayVersionString
  }

  fileprivate func updateGotAttention() {
    heldUpdate = nil
  }

  // MARK: - Rules

  /// Only a release checks for updates: a version from a release tag, a feed
  /// and a key in Info.plist, and a Team ID signature. A development
  /// build (0.0.0, `0.8.1-3-gabc1234`, `-dirty`) or an ad hoc one has nothing
  /// to update from, and a fork's unsigned build has no feed of its own.
  nonisolated static func isReleaseBuild(version: String, feedURL: String?, publicKey: String?, teamID: String?) -> Bool {
    guard version.wholeMatch(of: /\d+\.\d+\.\d+(-?(a|b|rc)\d+)?/) != nil, version != "0.0.0" else { return false }
    guard let feedURL, !feedURL.isEmpty, let publicKey, !publicKey.isEmpty else { return false }
    guard let teamID, !teamID.isEmpty else { return false }
    return true
  }

  /// The team of the running app's signature; nil when it is ad hoc.
  nonisolated static func signingTeam() -> String? {
    var code: SecCode?
    guard SecCodeCopySelf([], &code) == errSecSuccess, let code else { return nil }
    var staticCode: SecStaticCode?
    guard SecCodeCopyStaticCode(code, [], &staticCode) == errSecSuccess, let staticCode else { return nil }
    var info: CFDictionary?
    guard SecCodeCopySigningInformation(staticCode, SecCSFlags(rawValue: kSecCSSigningInformation), &info) == errSecSuccess,
      let info = info as? [String: Any]
    else { return nil }
    return info[kSecCodeInfoTeamIdentifier as String] as? String
  }

  /// The feed's notes are the changelog from the new release back several
  /// releases, each under a `Jetlink vX.Y.Z` heading. Cut at the installed
  /// release's heading so only what is new shows; nil keeps them all, when
  /// that heading is not there.
  nonisolated static func notesNewer(than installed: String, in notes: NSAttributedString) -> NSAttributedString? {
    let text = notes.string as NSString
    let heading = "Jetlink v\(installed)"
    var searchStart = 0
    while searchStart < text.length {
      let found = text.range(of: heading, range: NSRange(location: searchStart, length: text.length - searchStart))
      guard found.location != NSNotFound else { return nil }
      let line = text.lineRange(for: found)
      // the whole line, so v0.8.1 does not cut at v0.8.10
      if text.substring(with: line).trimmingCharacters(in: .whitespacesAndNewlines) == heading {
        guard line.location > 0 else { return nil }
        var end = line.location
        while end > 0, let scalar = UnicodeScalar(text.character(at: end - 1)), CharacterSet.whitespacesAndNewlines.contains(scalar) {
          end -= 1
        }
        return notes.attributedSubstring(from: NSRange(location: 0, length: end))
      }
      searchStart = NSMaxRange(found)
    }
    return nil
  }
}

/// Sparkle's delegates are Objective-C protocols, which need an NSObject. It
/// calls them on the main thread.
@MainActor
private final class UserDriverDelegate: NSObject, @preconcurrency SPUStandardUserDriverDelegate {
  weak var store: UpdateStore?

  var supportsGentleScheduledUpdateReminders: Bool { true }

  func standardUserDriverShouldHandleShowingScheduledUpdate(_ update: SUAppcastItem, andInImmediateFocus immediateFocus: Bool) -> Bool {
    store?.shouldSparkleShowScheduledUpdate() ?? true
  }

  func standardUserDriverWillHandleShowingUpdate(_ handleShowingUpdate: Bool, forUpdate update: SUAppcastItem, state: SPUUserUpdateState) {
    store?.willShowUpdate(update, handledBySparkle: handleShowingUpdate)
  }

  func standardUserDriverDidReceiveUserAttention(forUpdate update: SUAppcastItem) {
    store?.updateGotAttention()
  }

  func standardUserDriverWillFinishUpdateSession() {
    store?.updateGotAttention()
  }

  func standardUserDriverWillShowReleaseNotesText(
    _ releaseNotesAttributedString: NSAttributedString, forUpdate update: SUAppcastItem, withBundleDisplayVersion bundleDisplayVersion: String,
    bundleVersion: String
  ) -> NSAttributedString? {
    UpdateStore.notesNewer(than: bundleDisplayVersion, in: releaseNotesAttributedString)
  }
}
