package io.zoompilot.jetlink.server

import android.content.Context
import java.io.File

/**
 * The vendor's QNN 2.48.5 five, carried as assets and unpacked on first use.
 *
 * A V85 Hexagon opens one build and no other: the firmware's own
 * `v2.48.5.260901221701`, which carries a V85 pair alone and which no public
 * download has. The five cannot live in `jniLibs` beside the AAR's 2.42.0 set
 * -- two `libQnnHtp.so` of two builds under one name is the mix QNN swallows in
 * silence -- and the phone's own copy in `/odm/lib64` is a directory this app's
 * linker namespace does not answer for. So they travel as assets and are
 * unpacked to the app's own storage, which the namespace always reaches, and
 * the session is handed that directory as `qnn_library_dir`
 * [docs/qnn-htp-stubs.md].
 *
 * The Hexagon generation alone picks the set: every Hexagon up to V81 runs the
 * AAR's 2.42.0 from `jniLibs`, and only a V85 phone unpacks these.
 */
object QnnAssets {
    /** The build `/odm/lib64` carries on an SM8975, kept verbatim. */
    const val BUILD = "v2.48.5.260901221701"

    private const val DIRECTORY = "qnn-v85"

    /** Every name the set needs; a directory missing one is unpacked again. */
    private val FILES =
        listOf(
            "libQnnHtp.so",
            "libQnnHtpPrepare.so",
            "libQnnSystem.so",
            "libQnnHtpV85Stub.so",
            "libQnnHtpV85Skel.so",
        )

    /** Where the unpacked set lives: app storage, never backed up. */
    fun directory(context: Context): File = File(context.filesDir, "jetlink/qnn/$BUILD")

    /**
     * The directory holding the unpacked set, or null when this build carries
     * no assets for it. Idempotent: a directory the last run completed is
     * returned without touching the disk, and a partial one is replaced rather
     * than repaired -- a truncated 88 MB preparer would fail at the graph, not
     * at the copy.
     */
    fun install(context: Context): String? {
        val directory = directory(context)
        if (complete(directory)) return directory.absolutePath
        val names = runCatching { context.assets.list(DIRECTORY)?.toList() }.getOrNull().orEmpty()
        if (FILES.any { it !in names }) return null
        directory.deleteRecursively()
        if (!directory.mkdirs() && !directory.isDirectory) return null
        for (name in FILES) {
            val target = File(directory, name)
            val copied =
                runCatching {
                        context.assets.open("$DIRECTORY/$name").use { input ->
                            target.outputStream().use { output -> input.copyTo(output) }
                        }
                    }
                    .isSuccess
            if (!copied) {
                directory.deleteRecursively()
                return null
            }
        }
        // The stamp last: a directory is complete only once every file is.
        File(directory, STAMP).writeText(BUILD)
        return directory.absolutePath
    }

    private fun complete(directory: File): Boolean =
        File(directory, STAMP).let { it.isFile && it.readText().trim() == BUILD } &&
            FILES.all { File(directory, it).length() > 0 }

    private const val STAMP = ".unpacked"
}
