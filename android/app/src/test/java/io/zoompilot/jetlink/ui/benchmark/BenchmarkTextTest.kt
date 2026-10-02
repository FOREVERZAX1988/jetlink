package io.zoompilot.jetlink.ui.benchmark

import io.zoompilot.jetlink.R
import io.zoompilot.jetlink.server.Engine
import io.zoompilot.jetlink.server.RunState
import io.zoompilot.jetlink.settings.Chip
import io.zoompilot.jetlink.ui.PreviewData
import io.zoompilot.jetlink.ui.Tone
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Before
import org.junit.Test
import java.util.Locale

/** The Benchmark tab's blocker and commands. The note words compose, so they show in the UI, not here. */
class BenchmarkTextTest {
    private lateinit var locale: Locale

    @Before
    fun englishNumbers() {
        locale = Locale.getDefault()
        Locale.setDefault(Locale.US)
    }

    @After
    fun restore() {
        Locale.setDefault(locale)
    }

    @Test
    fun whatStopsARun() {
        assertEquals(R.string.bench_blocker_not_serving, BenchmarkText.blocker(RunState.Stopped, PreviewData.waiting))
        assertEquals(R.string.bench_blocker_no_model, BenchmarkText.blocker(RunState.Serving, PreviewData.waiting.copy(engine = Engine())))
        assertEquals(R.string.bench_blocker_comma, BenchmarkText.blocker(RunState.Serving, PreviewData.serving))
        assertNull(BenchmarkText.blocker(RunState.Serving, PreviewData.waiting))
        // a model still preparing is not loaded
        assertEquals(R.string.bench_blocker_no_model, BenchmarkText.blocker(RunState.Serving, PreviewData.preparing))
    }

    @Test
    fun theParityCommand() {
        val sha = PreviewData.BIG_MODEL_SHA
        val command = BenchmarkText.parityCommand(sha, 765_953_504, "192.168.1.23", 5599)
        assertEquals(
            "python3 scripts/verify_parity.py capture --host 192.168.1.23 --port 5599 --sha256 $sha --nbytes 765953504 --dir parity-android \\\n" +
                "  && python3 scripts/verify_parity.py reference --onnx \"\$HOME/Library/Application Support/Jetlink/cache/models/a086d5249fc308bb.onnx\" --dir parity-android \\\n" +
                "  && python3 scripts/verify_parity.py compare --dir parity-android",
            command,
        )
        assertNull(BenchmarkText.parityCommand(sha, 765_953_504, null, 5599))
        assertNull(BenchmarkText.parityCommand(null, 765_953_504, "192.168.1.23", 5599))
    }

    @Test
    fun theChip() {
        assertEquals("Snapdragon 8 Gen 3 · NPU v75", BenchmarkText.chipLine("Snapdragon 8 Gen 3", 75))
        assertEquals("SM7550", BenchmarkText.chipLine("SM7550", null))
        assertEquals(R.string.expectation_should to Tone.Good, BenchmarkText.expectation(Chip.Expectation.Recommended))
        assertEquals(Tone.Bad, BenchmarkText.expectation(Chip.Expectation.NoNpu).second)
    }
}
