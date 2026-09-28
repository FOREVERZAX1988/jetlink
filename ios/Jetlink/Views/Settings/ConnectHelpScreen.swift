import SwiftUI

/// How the comma and the phone meet: one cable through a hub. What
/// docs/iphone-app.md says, in the app, for a reader standing at the car.
struct ConnectHelpScreen: View {
  var body: some View {
    List {
      Section {
        step(1, "Set the comma to iOS.", "On the comma, parked: Accelerator Link, iOS. USB is for a Jetson or a Mac.")
        step(2, "Open Jetlink first.", "The phone dials the comma as soon as the cable is in. An app opened afterwards dials when it opens.")
        step(3, "Plug a USB 3 hub into the iPhone.", "One with power passthrough keeps the phone charged; the model runs 20 times a second.")
        step(4, "Join the hub to the comma with a USB 3 A-to-C cable.", "The A end goes in the hub, the C end in the comma.")
        step(5, "Wait for Connected over USB.", "The comma gives the phone an address over the cable. There is nothing to type.")
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
