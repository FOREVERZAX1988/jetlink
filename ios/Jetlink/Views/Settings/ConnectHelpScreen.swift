import SwiftUI

/// How the comma and the iPhone or iPad meet: one USB-C cable, or the
/// phone's hotspot, in a few steps for a reader standing at the car.
/// docs/iphone-app.md has the why.
struct ConnectHelpScreen: View {
  static let guide = URL(string: "https://github.com/zoompilot/jetlink/blob/main/docs/iphone-app.md#connect-the-comma")!

  var body: some View {
    let device = ThisDevice.name
    List {
      Section {
        step(1, "On the comma, set Jetlink to iOS.")
        step(2, "Open Jetlink and allow Local Network access.")
        step(3, "Connect your \(device) to the comma with a USB 3 USB-C cable.")
        step(4, "Wait for Connected.")
      } footer: {
        Text("A powered USB-C hub keeps your \(device) charging.")
      }
      Section {
        step(1, "In Jetlink's settings, turn on Wi-Fi Link.")
        step(2, "Turn on Personal Hotspot. Allow Others to Join on, Maximize Compatibility off.")
        step(3, "On the comma, join the hotspot in Network settings.")
        step(4, "On the comma, set Jetlink to Wi-Fi.")
        step(5, "Wait for Connected.")
      } header: {
        Text("Over Wi-Fi")
      } footer: {
        Text("No cable, but slower than USB. 2.4 GHz is too slow for a big model.")
      }
      Section {
        ForEach(USBSpeedGuide.rows, id: \.models) { row in
          LabeledContent(row.models, value: row.speed)
        }
      } header: {
        Text("Speed")
      } footer: {
        Text("Your iPhone or iPad needs a USB-C port. USB 2 works, but leaves less time for each frame.")
      }
      Section {
        Link("Learn More", destination: ConnectHelpScreen.guide)
      }
    }
    .readableWidth()
    .navigationTitle("Connecting the Comma")
    .navigationBarTitleDisplayMode(.inline)
  }

  private func step(_ number: Int, _ text: String) -> some View {
    HStack(alignment: .firstTextBaseline, spacing: 12) {
      Text(number.formatted())
        .font(.body.weight(.semibold).monospacedDigit())
        .foregroundStyle(.secondary)
        .frame(width: 18, alignment: .trailing)
      Text(text)
    }
    .accessibilityElement(children: .combine)
  }
}

#Preview {
  NavigationStack {
    ConnectHelpScreen()
  }
}
