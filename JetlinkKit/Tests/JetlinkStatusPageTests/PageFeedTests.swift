import Foundation
import JetlinkKit
import Testing

@testable import JetlinkStatusPage

@Suite("Status page feed")
struct PageFeedTests {
  let link = ControlEvent.link(LinkEvent(state: .connected, detail: "", peer: "usb", medium: "usb3"))

  func take(_ feed: PageFeed, _ stream: PageFeed.Stream) -> [ControlEvent] {
    (feed.next(stream, timeout: 0.01) ?? []).compactMap {
      if case .event(let event, _) = $0 { event } else { nil }
    }
  }

  @Test("An event is one data line, as jsonLine() writes it, then a blank line")
  func framing() throws {
    let line = link.jsonLine(at: Date(timeIntervalSince1970: 1_790_000_000.5))
    let frame = PageFeed.encode(.event(link, Date(timeIntervalSince1970: 1_790_000_000.5)))
    #expect(frame == Data("data: ".utf8) + line.dropLast() + Data("\n\n".utf8))
    #expect(!line.dropLast().contains(0x0A))
    let host = String(decoding: PageFeed.frame("host", ["hostname": "jetlink", "jetson": true], at: Date(timeIntervalSince1970: 2.5)), as: UTF8.self)
    #expect(host == #"data: {"event":"host","hostname":"jetlink","jetson":true,"t":2.5}"# + "\n\n")
    // What JSONSerialization cannot write is an error event, not a crash.
    let broken = String(decoding: PageFeed.frame("hw", ["when": Date()]), as: UTF8.self)
    #expect(broken.hasPrefix(#"data: {"error":"#) && broken.hasSuffix("\n\n"))
  }

  @Test("A page that opens late gets the latest of each and the last two minutes of stats")
  func replay() throws {
    let clock = ManualClock(1000)
    let feed = PageFeed(hardware: nil, clock: clock.read)
    feed.publish(.hello(HelloEvent(protocolVersion: 1, pid: 1, version: "0.7.0", python: "", platform: "linux", cache: "/c", transport: "usb", port: nil)))
    feed.publish(.server(ServerEvent(state: "serving", detail: "", backend: "trt", runtimeVersion: "10.16.2.10", device: "Orin")))
    feed.publish(.link(.waiting))
    feed.publish(.engine(.none))
    feed.publish(
      .inventory(InventoryEvent(loaded: nil, lastLoaded: nil, models: [], artifacts: [], disk: InventoryDisk(modelsBytes: 0, enginesBytes: 0, freeBytes: 0))))
    feed.publish(.catalog(CatalogEvent(fetchedAt: nil, url: "", defaultRef: "", error: nil, models: [])))
    feed.publish(.download(DownloadEvent(sha256: "a", ref: nil, state: "started", frac: 0, bytes: 0, total: 1, rateBps: 0, detail: "", source: nil)))
    feed.publish(stats(frames: 1))
    for (at, frames) in [(1030.0, 2), (1070, 3), (1110, 4), (1128, 5)] {
      feed.flush()
      clock.now = at
      feed.publish(stats(frames: frames))
    }
    feed.publish(link)
    feed.flush()
    clock.now = 1130

    let stream = try #require(feed.open())
    let replayed = take(feed, stream)
    #expect(replayed.map(\.name) == ["hello", "server", "link", "engine", "inventory", "stats", "stats", "stats", "stats"])
    #expect(replayed[2] == link)
    let frames = replayed.compactMap { if case .stats(let s) = $0 { s.frames } else { nil } }
    // the first stats came 130 s before the page: past the chart's two minutes
    #expect(frames == [2, 3, 4, 5])

    feed.publish(stats(frames: 6))
    feed.flush()
    #expect(take(feed, stream).map(\.name) == ["stats"])
  }

  @Test("The controller's state fills in only what no event has said")
  func seed() throws {
    let feed = PageFeed(hardware: nil)
    feed.publish(link)
    feed.seed([.server(ServerEvent(state: "serving", detail: "", backend: nil, runtimeVersion: nil, device: nil)), .link(.waiting), .engine(.none)])
    let stream = try #require(feed.open())
    let replayed = take(feed, stream)
    #expect(replayed.map(\.name) == ["server", "link", "engine"])
    #expect(replayed[1] == link)
  }

  @Test("At most eight pages at once")
  func cap() throws {
    let feed = PageFeed(hardware: nil)
    let open = (0..<8).compactMap { _ in feed.open() }
    #expect(open.count == 8)
    #expect(feed.open() == nil)
    feed.close(open[3])
    #expect(feed.open() != nil)
  }

  @Test("A page that falls too far behind is dropped, to reconnect")
  func behind() throws {
    let feed = PageFeed(hardware: nil, limits: PageFeed.Limits(behind: 4))
    let stream = try #require(feed.open())
    for n in 0..<5 { feed.publish(stats(frames: n)) }
    feed.flush()
    #expect(feed.next(stream, timeout: 0.01) == nil)
  }

  @Test("No sampling before a page opens; it stops a grace after the last one closes")
  func sampling() throws {
    let clock = ManualClock(0)
    let hardware = FakeHardware()
    let feed = PageFeed(hardware: hardware, limits: PageFeed.Limits(grace: 60, period: 0.01), clock: clock.read)
    feed.publish(link)
    feed.flush()
    Thread.sleep(forTimeInterval: 0.1)
    #expect(hardware.samples == 0 && hardware.hosts == 0)
    #expect(!feed.isSampling)

    let stream = try #require(feed.open())
    #expect(eventually { hardware.samples >= 3 })
    #expect(hardware.hosts == 1)
    let first = feed.next(stream, timeout: 1) ?? []
    #expect(first.contains { if case .frame(let data) = $0 { String(decoding: data, as: UTF8.self).contains(#""event":"host""#) } else { false } })

    clock.now = 10
    feed.close(stream)
    clock.now = 69
    let before = hardware.samples
    #expect(eventually { hardware.samples > before + 2 })
    #expect(feed.isSampling)

    clock.now = 71
    #expect(eventually { !feed.isSampling })
    let stopped = hardware.samples
    Thread.sleep(forTimeInterval: 0.1)
    #expect(hardware.samples == stopped)

    // A page after the pause: sampled again, the host not read again, and
    // the CPU counters forgotten so the first load is not a pause's average.
    let again = try #require(feed.open())
    #expect(eventually { hardware.samples > stopped })
    #expect(hardware.hosts == 1 && hardware.resets == 2)
    let replay = feed.next(again, timeout: 1) ?? []
    #expect(replay.contains { if case .frame(let data) = $0 { String(decoding: data, as: UTF8.self).contains(#""event":"host""#) } else { false } })
    feed.stop()
    #expect(eventually { !feed.isSampling })
  }

  // Lowered where the page is served: Linux and macOS.
  #if canImport(Darwin) || canImport(Glibc)
    @Test("The page's threads run at the lowest priority, and only they do")
    func priority() {
      final class Seen: @unchecked Sendable {
        let lock = NSLock()
        var value: Int32?
      }
      let seen = Seen()
      PageThread.start("jetlink-page-test") {
        #if os(Linux)
          let nice = getpriority(__priority_which_t(PRIO_PROCESS.rawValue), 0)
        #else
          let nice: Int32 = Thread.current.qualityOfService == .background ? 19 : 0
        #endif
        seen.lock.withLock { seen.value = nice }
      }
      #expect(eventually { seen.lock.withLock { seen.value } != nil })
      #expect(seen.lock.withLock { seen.value } == 19)
      #if os(Linux)
        #expect(getpriority(__priority_which_t(PRIO_PROCESS.rawValue), 0) != 19)
      #endif
    }
  #endif

  @Test("Stopping tells the open pages the server is stopping, then ends them")
  func stop() throws {
    let feed = PageFeed(hardware: nil)
    feed.publish(.server(ServerEvent(state: "serving", detail: "", backend: "trt", runtimeVersion: "10.16", device: "Orin")))
    feed.flush()
    let stream = try #require(feed.open())
    _ = feed.next(stream, timeout: 0.01)
    feed.stop()
    let last = take(feed, stream)
    #expect(last == [.server(ServerEvent(state: "stopping", detail: "", backend: "trt", runtimeVersion: "10.16", device: "Orin"))])
    #expect(feed.next(stream, timeout: 1) == nil)
    #expect(feed.open() == nil)
  }
}
