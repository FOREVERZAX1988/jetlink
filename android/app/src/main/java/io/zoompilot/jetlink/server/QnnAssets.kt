package io.zoompilot.jetlink.server

import android.content.Context
import android.util.Log
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
 * unpacked to the app's own storage, and the session is handed that directory
 * as `qnn_library_dir` [docs/qnn-htp-stubs.md].
 *
 * The Hexagon generation alone picks the set: every Hexagon up to V81 runs the
 * AAR's 2.42.0 from `jniLibs`, and only a V85 phone unpacks these.
 *
 * **They are made read-only once copied, and that is not tidiness.** From
 * Android 10 the linker refuses to open a library out of a directory the app
 * can still write to, and a file this app wrote is 0600 until told otherwise:
 * unpacked as-is, `dlopen` answers "is not accessible for the namespace" and
 * the session falls back to the AAR's build, which is the very silence this
 * whole arrangement exists to end. The stamp is written after that, so a
 * directory from a build that did not chmod is unpacked again rather than
 * trusted.
 */
object QnnAssets {
    /** The build `/odm/lib64` carries on an SM8975, kept verbatim. */
    const val BUILD = "v2.48.5.260901221701"

    private const val DIRECTORY = "qnn-v85"
    private const val TAG = "jetlink"

    /** Every name the set needs and the size it must land at, from `/odm/lib64`. */
    private val FILES =
        linkedMapOf(
            "libQnnHtp.so" to 3778744L,
            "libQnnHtpPrepare.so" to 88128192L,
            "libQnnSystem.so" to 4052904L,
            "libQnnHtpV85Stub.so" to 779448L,
            "libQnnHtpV85Skel.so" to 12582764L,
        )

    private const val STAMP = ".unpacked"

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
        if (FILES.keys.any { it !in names }) {
            Log.e(TAG, "the APK carries no $DIRECTORY assets: a V85 Hexagon has no set of its own to open")
            return null
        }
        directory.deleteRecursively()
        if (!directory.mkdirs() && !directory.isDirectory) return null
        for ((name, size) in FILES) {
            val target = File(directory, name)
            val copied =
                runCatching {
                    context.assets.open("$DIRECTORY/$name").use { input ->
                        target.outputStream().use { output -> input.copyTo(output) }
                    }
                    // The linker's rule, not the filesystem's: a library out of
                    // a writable file is a library it will not open.
                    target.setReadable(true, false)
                    target.setExecutable(true, false)
                    target.setWritable(false, false)
                }
                    .isSuccess
            if (!copied || target.length() != size) {
                Log.e(TAG, "could not unpack $name for the V85 Hexagon's QNN set (${target.length()} of $size bytes)")
                directory.deleteRecursively()
                return null
            }
        }
        // The stamp last: a directory is complete only once every file is, and
        // every file was made one the linker will open.
        File(directory, STAMP).writeText(BUILD)
        Log.i(TAG, "unpacked the V85 Hexagon's QNN $BUILD to $directory")
        return directory.absolutePath
    }

    /**
     * Complete means the stamp names this build, every file is the size the
     * vendor's is, and none is still writable. A copy killed halfway leaves
     * one the right length by chance and none that the loader would open, and
     * an 88 MB preparer that is short by a page fails at the graph, not here.
     */
    private fun complete(directory: File): Boolean =
        File(directory, STAMP).let { it.isFile && it.readText().trim() == BUILD } &&
            FILES.all { (name, size) ->
                File(directory, name).let { file -> file.length() == size && !file.canWrite() }
            }
}
