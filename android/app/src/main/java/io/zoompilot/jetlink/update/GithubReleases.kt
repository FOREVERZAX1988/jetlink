package io.zoompilot.jetlink.update

import kotlinx.serialization.json.Json
import java.net.HttpURLConnection
import java.net.URL

/**
 * Talks to the GitHub Releases API for [repo] ("owner/name"). The GitHub
 * release a [v* tag] publishes is the feed: [latest] fetches the newest
 * release, and [download] streams an asset to a file.
 *
 * The API is public and the fork's releases are public, so no token is
 * sent; a rate-limited answer reads as a failed check.
 */
class GithubReleases(
    val repo: String = "mouxangithub/jetlink",
    private val json: Json = Json { ignoreUnknownKeys = true },
    private val connect: (URL) -> HttpURLConnection = { it.openConnection() as HttpURLConnection },
) {
    /** The newest non-prerelease release, or null when there is none. */
    fun latest(): GithubRelease? {
        val url = URL("https://api.github.com/repos/$repo/releases/latest")
        val conn = connect(url)
        return try {
            conn.requestMethod = "GET"
            conn.setRequestProperty("Accept", "application/vnd.github+json")
            conn.setRequestProperty("User-Agent", "jetlink-android")
            conn.connectTimeout = 15_000
            conn.readTimeout = 15_000
            if (conn.responseCode != 200) return null
            conn.inputStream.bufferedReader().use { reader ->
                val release = json.decodeFromString(GithubRelease.serializer(), reader.readText())
                release.takeIf { !it.prerelease }
            }
        } finally {
            conn.disconnect()
        }
    }

    /** The Android APK asset of [release], the one the updater installs. */
    fun apk(release: GithubRelease): GithubAsset? =
        release.assets.firstOrNull { it.name.endsWith(".apk") && it.name.contains("Android", ignoreCase = true) }
            ?: release.assets.firstOrNull { it.name.endsWith(".apk") }
}

/**
 * Streams [url] to [outFile] with a progress callback, in the caller's
 * thread. Returns the number of bytes written, or throws on failure. A
 * redirect is followed once; GitHub's asset URLs redirect to objects.
 */
fun downloadToFile(
    url: String,
    outFile: java.io.File,
    connect: (URL) -> HttpURLConnection = { it.openConnection() as HttpURLConnection },
    onProgress: (bytes: Long, total: Long) -> Unit = { _, _ -> },
): Long {
    var current = URL(url)
    repeat(3) {
        val conn = connect(current)
        try {
            conn.requestMethod = "GET"
            conn.setRequestProperty("User-Agent", "jetlink-android")
            conn.connectTimeout = 20_000
            conn.readTimeout = 20_000
            when (conn.responseCode) {
                in 200..299 -> {
                    val total = conn.contentLengthLong
                    var written = 0L
                    val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                    conn.inputStream.use { input ->
                        outFile.outputStream().use { output ->
                            while (true) {
                                val read = input.read(buffer)
                                if (read < 0) break
                                output.write(buffer, 0, read)
                                written += read
                                onProgress(written, total)
                            }
                        }
                    }
                    return written
                }
                in 300..399 -> {
                    val location = conn.getHeaderField("Location") ?: return -1
                    current = URL(current, location)
                }
                else -> return -1
            }
        } finally {
            conn.disconnect()
        }
    }
    return -1
}