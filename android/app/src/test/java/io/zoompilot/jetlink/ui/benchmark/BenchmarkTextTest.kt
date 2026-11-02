package io.zoompilot.jetlink.ui.benchmark

import io.zoompilot.jetlink.R
import io.zoompilot.jetlink.server.Engine
import io.zoompilot.jetlink.server.RunState
import io.zoompilot.jetlink.settings.Chip
import io.zoompilot.jetlink.ui.PreviewData
import io.zoompilot.jetlink.ui.Tone
import io.zoompilot.jetlink.ui.Verdict
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Before
import org.junit.Test
import java.util.Locale

/** The Benchmark tab's blocker and chip words. The notes and the verdict's words compose, so they show in the UI, not here. */
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
    fun aRunWithNoFrames() {
        val none = PreviewData.report.copy(frames = 0, seconds = 178.0)
        assertEquals(Verdict.Slow, Verdict.of(none))
        val stopped = none.copy(cancelled = true)
        assertEquals(Verdict.None, Verdict.of(stopped))
        assertEquals(Verdict.Good, Verdict.of(PreviewData.report))
    }

    @Test
    fun theChip() {
        assertEquals("Snapdragon 8 Gen 3 · NPU v75", BenchmarkText.chipLine("Snapdragon 8 Gen 3", 75))
        assertEquals("SM7550", BenchmarkText.chipLine("SM7550", null))
        assertEquals(R.string.expectation_should to Tone.Good, BenchmarkText.expectation(Chip.Expectation.Recommended))
        assertEquals(R.string.expectation_unmeasured to Tone.Warning, BenchmarkText.expectation(Chip.Expectation.Unmeasured))
    }
}
