import Foundation
import JetlinkRegistry
import Testing

/// The URL rewrites the mirrors are tried in: a HuggingFace proxy swaps the
/// host, a GitHub prefix carries GitHub's hosts only, and the original URL
/// always closes the list. One test, because the configured bases are global.
@Suite("Mirrors")
struct MirrorTests {
  @Test("Candidates for every host, with mirrors configured and not")
  func candidates() {
    Mirrors.configure([])
    #expect(Mirrors.candidates(for: "https://raw.githubusercontent.com/commaai/openpilot/main/x.onnx").count == 1)

    Mirrors.configure(["https://hf-mirror.com", "https://ghfast.top"])

    // HuggingFace: the proxy swaps the host, then direct. A GitHub prefix
    // mirror does not carry HuggingFace's hosts.
    let hf = "https://huggingface.co/api/models/commaai/openpilot_driving_models/tree/main"
    #expect(
      Mirrors.candidates(for: hf) == [
        "https://hf-mirror.com/api/models/commaai/openpilot_driving_models/tree/main",
        hf,
      ],
    )
    // The LFS batch endpoint, a .git path, proxies the same way.
    let batch = "https://huggingface.co/commaai/openpilot-lfs.git/info/lfs"
    #expect(Mirrors.candidates(for: batch).first == "https://hf-mirror.com/commaai/openpilot-lfs.git/info/lfs")

    // GitHub: a prefix mirror carries it; the HuggingFace proxy cannot.
    let raw = "https://raw.githubusercontent.com/commaai/openpilot/main/openpilot/selfdrive/modeld/models/big_driving_supercombo.onnx"
    #expect(Mirrors.candidates(for: raw) == ["https://ghfast.top/\(raw)", raw])

    // GitLab has no mirror here: the original, alone.
    #expect(Mirrors.candidates(for: "https://gitlab.com/commaai/openpilot-lfs.git/info/lfs").count == 1)

    // A base that is not a URL is dropped at configure time.
    Mirrors.configure(["not a url", "https://hf-mirror.com"])
    #expect(Mirrors.configured() == ["https://hf-mirror.com"])

    Mirrors.configure([])
  }
}
