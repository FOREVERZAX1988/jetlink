import SwiftUI

/// How the comma and the iPhone or iPad meet: one USB-C cable, in a
/// few steps for a reader standing at the car. docs/iphone-app.md has the why.
struct ConnectHelpScreen: View {
  static let guide = URL(string: "https://github.com/zoompilot/jetlink/blob/main/docs/iphone-app.md#connect-the-comma")!

  var body: some View {
    let device = ThisDevice.name
    List {
      Section {
        step(1, "On the comma, set Accelerator Link to iOS.")
        step(2, "Open Jetlink and allow Local Network access.")
        step(3, "Connect your \(device) to the comma with a USB 3 USB-C cable.")
        step(4, "Wait for Connected.")
      } footer: {
        Text("A powered USB-C hub keeps your \(device) charging.")
      }
      // Apple's tech specs: USB 3 on the iPhone 15 Pro and later Pro models,
      // and on every iPad Pro, iPad Air and iPad mini with USB-C; USB 2 on
      // the other USB-C iPhones and on the iPad (10th generation) and (A16).
      Section {
        LabeledContent("iPhone 15 Pro and Later", value: "USB 3")
        LabeledContent("Other iPhones", value: "USB 2")
        LabeledContent("iPad Pro, Air and mini", value: "USB 3")
        LabeledContent("Other iPads", value: "USB 2")
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
