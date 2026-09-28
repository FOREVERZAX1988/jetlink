// The jetlink server as a command: the daemon on a Jetson or a Linux PC, and
// a Mac's server in a terminal. Only Linux and macOS build it; elsewhere this
// is an empty program, so the package still builds everything everywhere.
#if os(macOS) || os(Linux)
  import ArgumentParser
  import JetlinkKit

  struct JetlinkServerCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
      commandName: "jetlink-server",
      abstract: "Runs the comma's big model on this machine and serves it over USB or TCP.",
      version: Pinned.productVersion,
      subcommands: [Serve.self, ListBackends.self],
      defaultSubcommand: Serve.self)
  }

  JetlinkServerCommand.main()
#endif
