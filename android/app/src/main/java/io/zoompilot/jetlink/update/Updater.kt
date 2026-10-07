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

/** The updater's parsed result, for the About row. */
sealed interface UpdateState {
    /** No release found, or the check could not run. */
    data object Unknown : UpdateState
    /** The check is running. */
    data object Checking : UpdateState
    /** The installed version is the newest release. */
    data object Current : UpdateState
    /** [version] is newer than the installed one and its APK can be downloaded. */
    data class Available(val version: AppVersion) : UpdateState
    /** The check failed; [message] what went wrong. */
    data class Error(val message: String) : UpdateState
}