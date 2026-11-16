package io.zoompilot.jetlink.update

import kotlinx.serialization.json.Json
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Test

/** Parses the CNB releases payload a check answers. */
class CnbReleasesTest {
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
        val apk = CnbReleases().apk(release)
        assertNotNull(apk)
        assertEquals("Jetlink-0.8.3-Android.apk", apk?.name)
    }

    @Test
    fun parsesTheCnbPayloadWithItsExtraFields() {
        // the exact shape the anonymous https://cnb.cool/{repo}/-/releases
        // endpoint answers: extra fields the GitHub-era parser must ignore
        val payload = """
            [{
              "id": "2119302436095774720",
              "tag_name": "0.8.7-1108-1",
              "tag_commitish": "3acde9ccbaacab6b16207f20293785db7c759168",
              "name": "Jetlink cn 0.8.7-1108-1",
              "body": "Builds of the cn branch, for testing.",
              "draft": false,
              "prerelease": true,
              "is_latest": false,
              "created_at": "2026-11-08T05:34:10Z",
              "updated_at": "2026-11-08T05:34:10Z",
              "published_at": "2026-11-08T05:34:10Z",
              "author": {"username": "mouxan", "nickname": "x", "email": "x@noreply.cnb.cool", "is_npc": false},
              "assets": [{
                "id": "2119302533554626560",
                "name": "app-release.apk",
                "path": "/mouxancnb/jetlink/-/releases/download/0.8.7-1108-1/app-release.apk",
                "size": 40271360,
                "content_type": "application/vnd.android.package-archive",
                "browser_download_url": "https://cnb.cool/mouxancnb/jetlink/-/releases/download/0.8.7-1108-1/app-release.apk"
              }]
            }]
        """.trimIndent()
        val list = json.decodeFromString<List<GithubRelease>>(payload)
        assertEquals(1, list.size)
        assertEquals("0.8.7-1108-1", list[0].tag_name)
        val apk = CnbReleases().apk(list[0])
        assertNotNull(apk)
        assertEquals("app-release.apk", apk?.name)
        assertEquals(
            "https://cnb.cool/mouxancnb/jetlink/-/releases/download/0.8.7-1108-1/app-release.apk",
            apk?.browser_download_url,
        )
    }

    @Test
    fun rebuildsTheDownloadUrlWhenTheFeedOmitsIt() {
        val release = GithubRelease(
            tag_name = "0.8.7-1108-2",
            assets = listOf(GithubAsset("app-release.apk", "")),
        )
        assertEquals(
            "https://cnb.cool/mouxancnb/jetlink/-/releases/download/0.8.7-1108-2/app-release.apk",
            CnbReleases().apk(release)?.browser_download_url,
        )
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
        assertEquals("Jetlink-0.8.3-Android.apk", CnbReleases().apk(release)?.name)
    }

    @Test
    fun theFirstApkFallsBackWhenNoAndroidNamedOne() {
        val release = GithubRelease(
            tag_name = "v0.8.3",
            assets = listOf(GithubAsset("app-release.apk", "https://.../app.apk")),
        )
        assertEquals("app-release.apk", CnbReleases().apk(release)?.name)
    }

    @Test
    fun noApkMeansNoUpdate() {
        val release = GithubRelease(
            tag_name = "v0.8.3",
            assets = listOf(GithubAsset("Jetlink-0.8.3-macOS.zip", "https://.../mac.zip")),
        )
        assertNull(CnbReleases().apk(release))
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
        assertNotNull(CnbReleases().apk(list[0]))
        // the newest first: a cn release the updater should see
        assertEquals("Jetlink-android.apk", CnbReleases().apk(list[0])?.name)
    }
}
