import SwiftUI

/// How the comma and the phone meet: one cable through a hub, and the
/// Ethernet adapter as the fallback. What docs/iphone-app.md says, in
/// the app, for a reader standing at the car.
struct ConnectHelpScreen: View {
  var body: some View {
    List {
      Section {
        step(1, "Open Jetlink first.", "The phone dials the comma as soon as the cable is in. An app opened afterwards dials when it opens.")
        step(2, "Plug a USB 3 hub into the iPhone.", "One with power passthrough keeps the phone charged; the model runs 20 times a second.")
        step(3, "Join the hub to the comma with a USB 3 A-to-C cable.", "The A end goes in the hub, the C end in the comma.")
        step(4, "Wait for Connected over USB.", "The comma gives the phone an address over the cable. There is nothing to type.")
      } header: {
        Text("One Cable")
      } footer: {
        Text(
          "Do not plug the comma straight into the phone with a C-to-C cable. The two negotiate power, the comma ends up supplying the phone, and it reboots. Through a hub's A port the comma only ever draws."
        )
      }
      Section {
        Text(
          "Only the Pro iPhones, from the iPhone 15 Pro on, have a USB 3 port. The others, and the cable in the box, are USB 2. At USB 3 the cable costs a frame about 8 ms there and back, measured with a Mac standing in for the phone; USB 2 is expected to add a few milliseconds more. The title turns orange on USB 2. On the comma, the negotiated speed is in /sys/class/udc/*/current_speed: super-speed is USB 3, high-speed is USB 2."
        )
        .font(.subheadline)
        .foregroundStyle(.secondary)
      } header: {
        Text("USB 3")
      }
      Section {
        step(1, "Put an Ethernet adapter on each end.", "A USB-C gigabit adapter on the comma, one on the phone, and a short cable between them.")
        step(2, "Give the phone a manual address.", "Settings > Ethernet > the adapter > Configure IP > Manual, such as 10.0.0.2 with mask 255.255.255.0.")
        step(3, "Give the comma an address on the same subnet.", "Such as 10.0.0.1.")
        step(4, "Set the comma's JetlinkEndpoint.", "The Ethernet address this app shows under Connection, such as 10.0.0.2:5599. The comma dials the phone.")
      } header: {
        Text("Ethernet Adapter")
      } footer: {
        Text("The manual fallback, for a comma whose kernel has no USB network function. Wi-Fi works for testing and misses the frame budget.")
      }
    }
    .navigationTitle("Connecting the Comma")
    .navigationBarTitleDisplayMode(.inline)
  }

  private func step(_ number: Int, _ title: String, _ detail: String) -> some View {
    HStack(alignment: .top, spacing: 12) {
      Text(number.formatted())
        .font(.subheadline.weight(.semibold).monospacedDigit())
        .foregroundStyle(.secondary)
        .frame(width: 18, alignment: .trailing)
      VStack(alignment: .leading, spacing: 3) {
        Text(title)
          .font(.body.weight(.medium))
        Text(detail)
          .font(.subheadline)
          .foregroundStyle(.secondary)
      }
    }
    .padding(.vertical, 2)
  }
}

#Preview {
  NavigationStack {
    ConnectHelpScreen()
  }
}
