package io.zoompilot.jetlink.ui.benchmark

import androidx.compose.runtime.Composable
import io.zoompilot.jetlink.R
import io.zoompilot.jetlink.l10n
import io.zoompilot.jetlink.server.BenchReport
import io.zoompilot.jetlink.server.RunState
import io.zoompilot.jetlink.server.Snapshot
import io.zoompilot.jetlink.settings.Chip
import io.zoompilot.jetlink.ui.Format
import io.zoompilot.jetlink.ui.Thermal
import io.zoompilot.jetlink.ui.Tone
import io.zoompilot.jetlink.ui.Verdict
import io.zoompilot.jetlink.ui.status.StatusL10n

/** The Benchmark tab's words, as plain functions of the state. */
object BenchmarkText {
    /** Why a run cannot start now, as a resource; null when it can. The server refuses the same things. */
    fun blocker(serving: Boolean, modelLoaded: Boolean, commaConnected: Boolean): Int? = when {
        !serving -> R.string.bench_blocker_not_serving
        !modelLoaded -> R.string.bench_blocker_no_model
        commaConnected -> R.string.bench_blocker_comma
        else -> null
    }

    fun blocker(runState: RunState, snapshot: Snapshot): Int? =
        blocker(runState == RunState.Serving, loadedSha(snapshot) != null, snapshot.connected)

    /** The model a benchmark would run: the engine's, once it is ready. */
    fun loadedSha(snapshot: Snapshot): String? = snapshot.engine.sha256?.takeIf { snapshot.engine.state == "ready" }

    /** A benchmark that has not finished. */
    fun running(snapshot: Snapshot): Boolean = snapshot.benchmark?.state == "running"

    /** The line under the verdict: what it means, or that no frame finished. */
    @Composable
    fun verdictDetail(report: BenchReport, verdict: Verdict): String =
        if (report.frames == 0 && verdict == Verdict.Slow) {
            l10n(R.string.bench_verdict_no_frames, Format.clock(report.seconds))
        } else {
            verdictDetail(verdict)
        }

    /** "1:00 at 20 Hz", under the frame count. */
    @Composable
    fun framesNote(report: BenchReport): String = l10n(R.string.bench_frames_note, Format.clock(report.seconds))

    @Composable
    fun overNote(report: BenchReport): String =
        if (report.over35 > 0) l10n(R.string.bench_over_note_pos, Format.integer(report.over35)) else l10n(R.string.bench_over_note_none)

    /** "Throughout", or where the temperature started: "From normal". */
    @Composable
    fun thermalNote(report: BenchReport): String =
        if (report.thermalAtStart == report.thermalAtEnd) {
            l10n(R.string.bench_thermal_throughout)
        } else {
            l10n(R.string.bench_thermal_from, StatusL10n.thermalTitle(Thermal.of(report.thermalAtStart)))
        }

    /** "Snapdragon 8 Gen 3 · NPU v75". */
    fun chipLine(name: String, hexagon: Int?): String = listOfNotNull(name, hexagon?.let { "NPU v$it" }).joinToString(" · ")

    /** How well this phone should do, as a resource, before a benchmark says for sure. */
    fun expectation(expectation: Chip.Expectation): Pair<Int, Tone> = when (expectation) {
        Chip.Expectation.Recommended -> R.string.expectation_should to Tone.Good
        Chip.Expectation.Possible -> R.string.expectation_possible to Tone.Warning
        Chip.Expectation.TooOld -> R.string.expectation_too_old to Tone.Bad
        Chip.Expectation.Unmeasured -> R.string.expectation_unmeasured to Tone.Warning
    }

    @Composable
    fun verdictTitle(verdict: Verdict): String = l10n(
        when (verdict) {
            Verdict.Good -> R.string.verdict_good_title
            Verdict.Tight -> R.string.verdict_tight_title
            Verdict.Slow -> R.string.verdict_slow_title
            Verdict.None -> R.string.verdict_none_title
        },
    )

    @Composable
    fun verdictDetail(verdict: Verdict): String = l10n(
        when (verdict) {
            Verdict.Good -> R.string.verdict_good_detail
            Verdict.Tight -> R.string.verdict_tight_detail
            Verdict.Slow -> R.string.verdict_slow_detail
            Verdict.None -> R.string.verdict_none_detail
        },
    )
}
