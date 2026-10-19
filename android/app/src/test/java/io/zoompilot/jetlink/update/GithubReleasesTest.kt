package io.zoompilot.jetlink.update

import kotlinx.serialization.json.Json
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Test

/** Parses the GitHub Releases payload a check answers. */
class GithubReleasesTest {
    private val json = Json { ignoreUnknownKeys = true }

    @Test
    fun parsesTheLatestReleasePayload() {
        val payload = """
            {
              "tag_name": "v0.8.3",
              "prerelease": false,
              "assets": [
                {"name": "Jetlink-0.8.3-Android.apk", "browser_download_url": "https://github.com/.../Jetlink-0.8.3-Android.apk"},
                {"name": "Jetlink-0.8.3-macOS.zip", "browser_download_url": "https://github.com/.../Jetlink-0.8.3-macOS.zip"}
              ]
            }
        """.trimIndent()
        val release = json.decodeFromString(GithubRelease.serializer(), payload)
        assertEquals("v0.8.3", release.tag_name)
        assertEquals(2, release.assets.size)
        val apk = GithubReleases().apk(release)
        assertNotNull(apk)
        assertEquals("Jetlink-0.8.3-Android.apk", apk?.name)
    }

    @Test
    fun anAndroidApkIsChosenAmongTheAssets() {
        val release = GithubRelease(
            tag_name = "v0.8.3",
            assets = listOf(
                GithubAsset("Jetlink-0.8.3-macOS.zip", "https://.../mac.zip"),
                GithubAsset("Jetlink-0.8.3-Android.apk", "https://.../app.apk"),
            ),
        )
        assertEquals("Jetlink-0.8.3-Android.apk", GithubReleases().apk(release)?.name)
    }

    @Test
    fun theFirstApkFallsBackWhenNoAndroidNamedOne() {
        val release = GithubRelease(
            tag_name = "v0.8.3",
            assets = listOf(GithubAsset("app-release.apk", "https://.../app.apk")),
        )
        assertEquals("app-release.apk", GithubReleases().apk(release)?.name)
    }

    @Test
    fun noApkMeansNoUpdate() {
        val release = GithubRelease(
            tag_name = "v0.8.3",
            assets = listOf(GithubAsset("Jetlink-0.8.3-macOS.zip", "https://.../mac.zip")),
        )
        assertNull(GithubReleases().apk(release))
    }

    @Test
    fun parsesTheReleasesListWithPrereleases() {
        val payload = """
            [
              {"tag_name": "cn-abc1234", "prerelease": true, "assets": [
                {"name": "Jetlink-android.apk", "browser_download_url": "https://.../cn.apk"}
              ]},
              {"tag_name": "v0.8.3", "prerelease": false, "assets": [
                {"name": "Jetlink-0.8.3-Android.apk", "browser_download_url": "https://.../v083.apk"}
              ]}
            ]
        """.trimIndent()
        val list = json.decodeFromString<List<GithubRelease>>(payload)
        assertEquals(2, list.size)
        assertEquals("cn-abc1234", list[0].tag_name)
        assertNotNull(GithubReleases().apk(list[0]))
        // the newest first: a cn prerelease the updater should see
        assertEquals("Jetlink-android.apk", GithubReleases().apk(list[0])?.name)
    }
}