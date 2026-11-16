package io.zoompilot.jetlink.server

import android.content.Context
import android.util.Log
import java.io.File

/**
 * libavcodec/libavutil, carried as assets and loaded before libjetlink.
 *
 * The Swift server's HEVC decoder (CHevc/jl_hevc.c) opens libavcodec at run
 * time with plain `dlopen("libavcodec.so")`. Android has no system libavcodec,
 * so a stock APK reports no HEVC in its hello and the comma falls back to
 * lossless frames — ~390 KB each, past what a phone's Wi-Fi carries at 20 Hz,
 * which is what demotes the large model. The FFmpeg libraries are LGPL and
 * arm64-generic: unlike the vendor's QNN set they are not tied to a Hexagon
 * generation, so one copy works on the 8 Elite (SM8975) and older phones.
 *
 * They cannot live in jniLibs: AGP would extract them beside libjetlink.so
 * only at install, and the two extra module names need no build wiring. As
 * assets they are unpacked once to app storage and loaded explicitly. Loading
 * by absolute path registers their SONAME with the linker, so the later
 * `dlopen("libavcodec.so")` from inside libjetlink resolves to these copies —
 * without touching the decoder's C code.
 *
 * As with QNN: Android 10+ refuses to dlopen a library from a writable file,
 * so the copies are made read-only + executable before they are loaded, and
 * the stamp is written last.
 */
object AvAssets {
    private const val DIRECTORY = "ffmpeg-arm64"
    private const val TAG = "jetlink"

    /** Dependency order: avcodec needs swresample + avutil, swresample avutil. */
    private val FILES = listOf("libavutil.so", "libswresample.so", "libavcodec.so")

    private const val STAMP = ".loaded"

    /** Where the unpacked libraries live: app storage, never backed up. */
    private fun directory(context: Context): File = File(context.filesDir, "jetlink/ffmpeg")

    /**
     * Unpack and load the HEVC decoder libraries. Idempotent and best effort:
     * returns the directory when loaded, null when this build carries no
     * assets (an older APK, or a device that cannot load them). A failure
     * leaves the server on its lossless carrier, which is the old behavior.
     */
    fun install(context: Context): String? {
        val directory = directory(context)
        val stamp = File(directory, STAMP)
        if (stamp.isFile) return directory.absolutePath

        val names = runCatching { context.assets.list(DIRECTORY)?.toList() }.getOrNull().orEmpty()
        if (FILES.any { it !in names }) {
            Log.w(TAG, "the APK carries no $DIRECTORY assets: HEVC stays unavailable")
            return null
        }

        runCatching {
            directory.deleteRecursively()
            if (!directory.mkdirs() && !directory.isDirectory) return null
            for (name in FILES) {
                val target = File(directory, name)
                context.assets.open("$DIRECTORY/$name").use { input ->
                    target.outputStream().use { output -> input.copyTo(output) }
                }
                // The linker's rule: a library in a writable file is one it
                // will not open.
                target.setReadable(true, false)
                target.setExecutable(true, false)
                target.setWritable(false, false)
            }
            // Load in dependency order; a failure here is why the subsequent
            // dlopen from libjetlink would report no decoder.
            for (name in FILES) System.load(File(directory, name).absolutePath)
            stamp.writeText("arm64")
            Log.i(TAG, "loaded HEVC decoder libraries from $directory")
        }.onFailure { error ->
            Log.e(TAG, "could not load the HEVC decoder libraries: ${error.message}")
            directory.deleteRecursively()
            return null
        }
        return directory.absolutePath
    }
}
