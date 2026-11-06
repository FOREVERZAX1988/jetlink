package io.zoompilot.jetlink.update

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

/**
 * What the GitHub Releases API says about a release, the fields the
 * updater reads. Field names are the API's snake_case.
 */
@Serializable
data class GithubRelease(
    val tag_name: String = "",
    val prerelease: Boolean = false,
    val assets: List<GithubAsset> = emptyList(),
)

@Serializable
data class GithubAsset(
    val name: String = "",
    val browser_download_url: String = "",
)

/** A version from a "v0.8.3"-shaped tag, compared numerically. */
data class AppVersion(val major: Int, val minor: Int, val patch: Int) : Comparable<AppVersion> {
    override fun compareTo(other: AppVersion): Int =
        compareValuesBy(this, other, { it.major }, { it.minor }, { it.patch })

    override fun toString(): String = "$major.$minor.$patch"

    companion object {
        /** Parses "0.8.3", "v0.8.3", "v0.8.3-rc1"; null when it is not a version. */
        fun parse(text: String): AppVersion? {
            val core = text.trim().removePrefix("v").substringBefore('-')
            val parts = core.split('.')
            if (parts.size < 3) return null
            val numbers = parts.map { it.toIntOrNull() ?: return null }
            return AppVersion(numbers[0], numbers[1], numbers[2])
        }
    }
}

/**
 * A build the updater can order: the app version, then the calendar day
 * (MMDD), then that day's build number - exactly the tag CI generates:
 * "0.8.7-1015-1" is 0.8.7, October 15, the first build; "-4" the fourth; the
 * next day restarts at "-1016-1", and a version bump (0.8.8) sorts above an
 * older 0.8.7 regardless of the day.
 *
 * A plain release tag ("v0.8.7" / "0.8.7") orders as that version with day
 * and build 0, so it is older than any dated prerelease of the same version
 * but newer than an older version's prereleases. Unparseable tags ("local",
 * "cn-<sha>") return null and never trigger an update.
 */
data class BuildTag(
    val version: AppVersion,
    val day: Int,
    val build: Int,
) : Comparable<BuildTag> {
    override fun compareTo(other: BuildTag): Int =
        compareValuesBy(this, other, { it.version }, { it.day }, { it.build })

    override fun toString(): String =
        if (day == 0) "$version" else "$version-%04d-%d".format(day, build)

    companion object {
        private val DATED = Regex("""^v?(\d+)\.(\d+)\.(\d+)-(\d{4})-(\d+)$""")
        // a bare release tag, with no prerelease suffix: "0.8.7" or "v0.8.7"
        private val BARE = Regex("""^v?(\d+)\.(\d+)\.(\d+)$""")

        fun parse(text: String): BuildTag? {
            val tag = text.trim()
            DATED.matchEntire(tag)?.let { m ->
                val (mj, mn, pt, day, n) = m.destructured
                return BuildTag(AppVersion(mj.toInt(), mn.toInt(), pt.toInt()), day.toInt(), n.toInt())
            }
            // a bare release tag: same version, before any dated build of it
            BARE.matchEntire(tag)?.let { m ->
                val (mj, mn, pt) = m.destructured
                return BuildTag(AppVersion(mj.toInt(), mn.toInt(), pt.toInt()), 0, 0)
            }
            return null
        }
    }
}

/** The updater's parsed result, for the About row. */
sealed interface UpdateState {
    /** No release found, or the check could not run. */
    data object Unknown : UpdateState
    /** The check is running. */
    data object Checking : UpdateState
    /** The installed build is the newest release. */
    data object Current : UpdateState
    /** [tag] is newer than the installed one and its APK can be downloaded. */
    data class Available(val tag: String) : UpdateState
    /** The check failed; [message] what went wrong. */
    data class Error(val message: String) : UpdateState
}