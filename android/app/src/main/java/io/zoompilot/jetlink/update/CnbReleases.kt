package io.zoompilot.jetlink.update

import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import java.net.HttpURLConnection
import java.net.URL

/**
 * Talks to the CNB releases feed for [repo] ("owner/name" slug). CI uploads
 * every cn-branch APK as a CNB release asset (scripts/cnb-publish-release.sh,
 * called from .github/workflows/ci.yml); this class is the phone-side half.
 * [list] fetches the newest releases (prereleases included) from the public
 * web endpoint, which answers JSON without a token; its fields keep the
 * GitHub Releases API names (tag_name, prerelease, assets[].browser_download_url).
 *
 * A rate limit or an unreachable feed reads as a failed check, never a
 * downgrade.
 */
class CnbReleases(
    val repo: String = "mouxancnb/jetlink",
    private val json: Json = Json { ignoreUnknownKeys = true },
    private val connect: (URL) -> HttpURLConnection = { it.openConnection() as HttpURLConnection },
) {
    /**
     * The newest [perPage] releases. Unlike the "latest" endpoint the list
     * sees the cn-branch builds, which is what the updater compares its
     * build tag with.
     */
    fun list(perPage: Int = 5): List<GithubRelease> {
        val url = URL("https://cnb.cool/$repo/-/releases?per_page=$perPage")
        val conn = connect(url)
        return try {
            conn.requestMethod = "GET"
            conn.setRequestProperty("Accept", "application/json")
            conn.setRequestProperty("User-Agent", "jetlink-android")
            conn.connectTimeout = 15_000
            conn.readTimeout = 15_000
            if (conn.responseCode != 200) return emptyList()
            conn.inputStream.bufferedReader().use { reader ->
                json.decodeFromString<List<GithubRelease>>(reader.readText())
            }
        } finally {
            conn.disconnect()
        }
    }

    /**
     * The Android APK asset of [release], the one the updater installs.
     * CNB serves release assets at a fixed download path; when the feed
     * omits the URL the asset's is rebuilt from that rule.
     */
    fun apk(release: GithubRelease): GithubAsset? {
        val asset = release.assets.firstOrNull { it.name.endsWith(".apk") && it.name.contains("Android", ignoreCase = true) }
            ?: release.assets.firstOrNull { it.name.endsWith(".apk") } ?: return null
        if (asset.browser_download_url.isBlank() && release.tag_name.isNotBlank() && asset.name.isNotBlank()) {
            return asset.copy(
                browser_download_url =
                "https://cnb.cool/$repo/-/releases/download/${release.tag_name}/${asset.name}",
            )
        }
        return asset
    }
}

/**
 * Streams [url] to [outFile] with a progress callback, in the caller's
 * thread. Returns the number of bytes written, or throws on failure. A
 * redirect is followed once; CNB's asset URLs redirect to the object store.
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
