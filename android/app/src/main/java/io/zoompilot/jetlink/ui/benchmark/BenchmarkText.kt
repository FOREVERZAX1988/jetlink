package io.zoompilot.jetlink.ui.benchmark

import io.zoompilot.jetlink.server.BenchReport
import io.zoompilot.jetlink.server.RunState
import io.zoompilot.jetlink.server.Snapshot
import io.zoompilot.jetlink.settings.Chip
import io.zoompilot.jetlink.ui.Format
import io.zoompilot.jetlink.ui.Thermal
import io.zoompilot.jetlink.ui.Tone

/** The Benchmark tab's words and commands, as plain functions of the state. */
object BenchmarkText {
    /** The live bench on the comma: its cameras and modeld, over this phone's link. */
    const val COMMA_COMMAND = "/data/openpilot/jetlink_repo/scripts/comma/jetlink_live_bench.sh 180"

    const val GUIDE = "https://github.com/zoompilot/jetlink/blob/main/docs/android-app.md#benchmark"

    /** Why a run cannot start now, in a sentence; null when it can. The server refuses the same things. */
    fun blocker(serving: Boolean, modelLoaded: Boolean, commaConnected: Boolean): String? = when {
        !serving -> "The server is not running."
        !modelLoaded -> "Load a model first."
        commaConnected -> "Disconnect the comma first."
        else -> null
    }

    fun blocker(runState: RunState, snapshot: Snapshot): String? =
        blocker(runState == RunState.Serving, loadedSha(snapshot) != null, snapshot.connected)

    /** The model a benchmark would run: the engine's, once it is ready. */
    fun loadedSha(snapshot: Snapshot): String? = snapshot.engine.sha256?.takeIf { snapshot.engine.state == "ready" }

    /** A benchmark that has not finished. */
    fun running(snapshot: Snapshot): Boolean = snapshot.benchmark?.state == "running"

    /**
     * verify_parity from a Mac on the same Wi-Fi, dialing the phone's listener,
     * then checking its outputs against onnxruntime on the Mac.
     */
    fun parityCommand(sha256: String?, bytes: Long?, host: String?, port: Int?): String? {
        if (sha256 == null || bytes == null || host == null || port == null) return null
        val onnx = "\"\$HOME/Library/Application Support/Jetlink/cache/models/${sha256.take(16)}.onnx\""
        return "python3 scripts/verify_parity.py capture --host $host --port $port --sha256 $sha256 --nbytes $bytes --dir parity-android \\\n" +
            "  && python3 scripts/verify_parity.py reference --onnx $onnx --dir parity-android \\\n" +
            "  && python3 scripts/verify_parity.py compare --dir parity-android"
    }

    /** "1:00 at 20 Hz", under the frame count. */
    fun framesNote(report: BenchReport): String = "${Format.clock(report.seconds)} at 20 Hz"

    fun overNote(report: BenchReport): String =
        if (report.over35 > 0) "${Format.integer(report.over35)} over 35 ms" else "None over 35 ms"

    /** "Throughout", or where the temperature started: "From normal". */
    fun thermalNote(report: BenchReport): String =
        if (report.thermalAtStart == report.thermalAtEnd) "Throughout" else "From ${Thermal.of(report.thermalAtStart).title.lowercase()}"

    /** "Snapdragon 8 Gen 3 · NPU v75". */
    fun chipLine(name: String, hexagon: Int?): String = listOfNotNull(name, hexagon?.let { "NPU v$it" }).joinToString(" · ")

    /** How well this phone should do, before a benchmark says for sure. */
    fun expectation(expectation: Chip.Expectation): Pair<String, Tone> = when (expectation) {
        Chip.Expectation.Recommended -> "Should keep up" to Tone.Good
        Chip.Expectation.Possible -> "Might keep up" to Tone.Warning
        Chip.Expectation.TooOld -> "NPU too old" to Tone.Bad
        Chip.Expectation.NoNpu -> "No Snapdragon NPU" to Tone.Bad
    }
}
