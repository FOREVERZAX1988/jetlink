import Foundation
import Testing

@testable import Jetlink

struct EmbeddedPythonTests {
  /// Bundle.resourceURL comes back relative to the bundle, like this one. On
  /// macOS 15 a relative URL's path(percentEncoded:) is just the relative
  /// part, so the interpreter has to be resolved before it is checked or run.
  private let resources = URL(string: "Contents/Resources/", relativeTo: URL(filePath: "/Applications/Jetlink.app/"))!

  @Test func theBundledInterpreterIsAbsolute() {
    let interpreter = PythonRuntime.bundledInterpreter(resources: resources)
    #expect(interpreter.baseURL == nil)
    #expect(interpreter.relativePath == "/Applications/Jetlink.app/Contents/Resources/python/bin/python3")
    #expect(interpreter.path(percentEncoded: false) == "/Applications/Jetlink.app/Contents/Resources/python/bin/python3")
  }
}
