import AppKit
import Testing

@testable import Jetlink

@Suite("Updates")
struct UpdateStoreTests {
  private let feed = "https://github.com/zoompilot/jetlink/releases/latest/download/appcast.xml"
  private let key = "Fy8Q4q7tZqdU91t94/uVb6M5aeHp9z3oOLsfO4/NNY4="

  @Test("A signed release with a feed and a key checks for updates")
  func releaseBuild() {
    for version in ["0.8.1", "1.0.0", "0.9.0rc1", "0.9.0b2", "0.7.0-rc1"] {
      #expect(UpdateStore.isReleaseBuild(version: version, feedURL: feed, publicKey: key, teamID: "7Y92PH4BPZ"), "\(version)")
    }
  }

  @Test("Development builds, ad hoc ones and ones without a feed or key do not")
  func notReleaseBuild() {
    for version in ["0.0.0", "0.8.1-4-gcba9f60", "0.8.1-4-gcba9f60-dirty", "0.8.1-dirty", "cba9f60", ""] {
      #expect(!UpdateStore.isReleaseBuild(version: version, feedURL: feed, publicKey: key, teamID: "7Y92PH4BPZ"), "\(version)")
    }
    #expect(!UpdateStore.isReleaseBuild(version: "0.8.1", feedURL: feed, publicKey: key, teamID: nil))
    #expect(!UpdateStore.isReleaseBuild(version: "0.8.1", feedURL: feed, publicKey: key, teamID: ""))
    #expect(!UpdateStore.isReleaseBuild(version: "0.8.1", feedURL: "", publicKey: key, teamID: "7Y92PH4BPZ"))
    #expect(!UpdateStore.isReleaseBuild(version: "0.8.1", feedURL: feed, publicKey: nil, teamID: "7Y92PH4BPZ"))
  }

  private let notes = NSAttributedString(string: "Jetlink v0.9.0\nNew in 0.9.0\n\nJetlink v0.8.10\nNew in 0.8.10\n\nJetlink v0.8.1\nNew in 0.8.1\n")

  @Test("The notes stop at the installed release")
  func notesCut() {
    #expect(UpdateStore.notesNewer(than: "0.8.1", in: notes)?.string == "Jetlink v0.9.0\nNew in 0.9.0\n\nJetlink v0.8.10\nNew in 0.8.10")
    #expect(UpdateStore.notesNewer(than: "0.8.10", in: notes)?.string == "Jetlink v0.9.0\nNew in 0.9.0")
  }

  @Test("Notes without the installed release, or starting with it, stay whole")
  func notesKept() {
    #expect(UpdateStore.notesNewer(than: "0.7.0", in: notes) == nil)
    #expect(UpdateStore.notesNewer(than: "0.9.0", in: notes) == nil)
    #expect(UpdateStore.notesNewer(than: "0.8", in: notes) == nil)
  }
}
