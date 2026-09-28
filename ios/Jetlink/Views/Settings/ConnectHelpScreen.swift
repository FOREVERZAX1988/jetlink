import SwiftUI

/// How the comma and the phone meet: one cable through a hub, in a few
/// steps for a reader standing at the car. docs/iphone-app.md has the why.
struct ConnectHelpScreen: View {
  static let guide = URL(string: "https://github.com/zoompilot/jetlink/blob/main/docs/iphone-app.md#connect-the-comma")!

  var body: some View {
    List {
      Section {
        step(1, "On the comma, set Accelerator Link to iOS.")
        step(2, "Open Jetlink and allow Local Network access.")
        step(3, "Plug a USB 3 hub into your iPhone.")
        step(4, "Connect the hub to the comma with a USB 3 A-to-C cable.")
        step(5, "Wait for Connected.")
      } footer: {
        Text("Don't connect the comma straight to your iPhone with a USB-C cable.")
      }
      Section {
        LabeledContent("iPhone 15 Pro and Later", value: "USB 3")
        LabeledContent("Other iPhones", value: "USB 2")
      } header: {
        Text("Speed")
      } footer: {
        Text("USB 2 works, but leaves less time for each frame.")
      }
      Section {
        Link("Learn More", destination: ConnectHelpScreen.guide)
      }
    }
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
