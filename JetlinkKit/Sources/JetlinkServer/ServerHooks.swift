import Foundation

/// What the USB loop and the sessions tell a host that suspends while the
/// comma is gone: the Python serve loop's `touch` and `idle` calls.
public enum GadgetIdleEvent: Sendable, Equatable {
  /// The gadget is on the bus, whether or not it opens. A comma whose gadget
  /// cannot be opened yet is still there, and the box must not sleep under it.
  case present
  /// No gadget on the bus and no comma being served, over any link: the only
  /// time the host may go to sleep.
  case absent
  /// A comma connected, over any link.
  case connected
  /// That connection ended.
  case disconnected
}

/// What the host running the server supplies besides the backend and the
/// gadget. Every hook defaults to what the apps do today, so they pass none;
/// the Linux daemon fills them in.
public struct ServerHooks: Sendable {
  /// The host's sensors, for a frame that asks for them (WANT_STATE), the
  /// hello and STATE_RESP: `{}` when it has none, never zeros. The server
  /// reads it on a thread of its own (`TelemetrySampler`), so it may block.
  public var telemetry: @Sendable () -> [String: Any]
  /// The device's thermal state for the benchmark's reports: "nominal",
  /// "fair", "serious" or "critical". The platform's own unless the host
  /// knows better (Android's PowerManager).
  public var thermal: @Sendable () -> String
  /// HELLO_RESP's `sleep_after`: the seconds with no gadget after which this
  /// host suspends. 0 says it never does, and the comma then holds the gadget
  /// for the whole park. Only a host that really sleeps may say more.
  public var sleepAfter: Double
  /// Hears the gadget's presence on every USB poll and every connection's
  /// start and end. After `.absent` it may suspend the host, returning once
  /// the host is awake again; it returns true when it slept, and the loop
  /// then looks for the gadget at once, since whatever woke the box is
  /// likely the comma. The return value means nothing for the other events.
  public var gadgetIdle: (@Sendable (GadgetIdleEvent) -> Bool)?
  /// A SHUTDOWN_REQ, with the comma's reason. nil, the default, refuses: the
  /// reply says ok:false and the apps hear `.shutdownRequested`, as a phone
  /// or a Mac does not power itself off for the comma. The hook refuses too
  /// by returning nil, or accepts by returning what to do: the reply says
  /// `{ok: true, detail: "powering off"}`, and the action runs only once the
  /// reply is written, so the comma has its answer before the host goes down.
  public var shutdown: (@Sendable (_ reason: String) -> (@Sendable () -> Void)?)?
  /// Called on the session thread after a frame was answered INFER_FAILED
  /// for an error the backend marks fatal (`FatalEngineError`). The daemon
  /// exits with status 3 and systemd starts it again; an app does nothing,
  /// and the next frame fails the same way.
  public var fatal: (@Sendable (any Error) -> Void)?

  public init(
    telemetry: @escaping @Sendable () -> [String: Any] = { [:] },
    thermal: @escaping @Sendable () -> String = { platformThermal() },
    sleepAfter: Double = 0,
    gadgetIdle: (@Sendable (GadgetIdleEvent) -> Bool)? = nil,
    shutdown: (@Sendable (_ reason: String) -> (@Sendable () -> Void)?)? = nil,
    fatal: (@Sendable (any Error) -> Void)? = nil
  ) {
    self.telemetry = telemetry
    self.thermal = thermal
    self.sleepAfter = sleepAfter
    self.gadgetIdle = gadgetIdle
    self.shutdown = shutdown
    self.fatal = fatal
  }
}
