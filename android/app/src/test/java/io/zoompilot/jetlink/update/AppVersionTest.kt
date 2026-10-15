package io.zoompilot.jetlink.update

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** The version comparison the updater decides an update by. */
class AppVersionTest {
    @Test
    fun parsesTheTagShapes() {
        assertEquals(AppVersion(0, 8, 3), AppVersion.parse("v0.8.3"))
        assertEquals(AppVersion(0, 8, 3), AppVersion.parse("0.8.3"))
        assertEquals(AppVersion(0, 8, 3), AppVersion.parse("v0.8.3-rc1"))
        assertEquals(AppVersion(1, 0, 0), AppVersion.parse("v1.0.0"))
    }

    @Test
    fun rejectsWhatIsNotAVersion() {
        assertNull(AppVersion.parse(""))
        assertNull(AppVersion.parse("v0.8"))
        assertNull(AppVersion.parse("latest"))
        assertNull(AppVersion.parse("v0.8.x"))
    }

    @Test
    fun comparesNumericallyNotLexicographically() {
        assertTrue(AppVersion(0, 8, 3) < AppVersion(0, 8, 4))
        assertTrue(AppVersion(0, 8, 3) < AppVersion(0, 9, 0))
        assertTrue(AppVersion(0, 8, 3) < AppVersion(1, 0, 0))
        assertTrue(AppVersion(0, 8, 10) > AppVersion(0, 8, 9))
        assertEquals(0, AppVersion(0, 8, 3).compareTo(AppVersion(0, 8, 3)))
    }

    @Test
    fun formatsWithoutTheLetter() {
        assertEquals("0.8.3", AppVersion(0, 8, 3).toString())
    }

    // -- BuildTag: "<version>-<MMDD>-<n>", the cn prerelease ordering -----

    @Test
    fun parsesDatedBuildTags() {
        assertEquals(BuildTag(AppVersion(0, 8, 7), 1015, 1), BuildTag.parse("0.8.7-1015-1"))
        assertEquals(BuildTag(AppVersion(0, 8, 7), 1015, 4), BuildTag.parse("0.8.7-1015-4"))
        assertEquals(BuildTag(AppVersion(0, 8, 8), 1016, 1), BuildTag.parse("0.8.8-1016-1"))
        // a bare release tag orders before any dated build of the same version
        assertEquals(BuildTag(AppVersion(0, 8, 7), 0, 0), BuildTag.parse("v0.8.7"))
    }

    @Test
    fun rejectsTagsThatAreNotBuilds() {
        assertNull(BuildTag.parse(""))
        assertNull(BuildTag.parse("local"))
        assertNull(BuildTag.parse("cn-abc1234"))
        assertNull(BuildTag.parse("0.8.7-1015"))        // missing build number
        assertNull(BuildTag.parse("0.8.7-15-1"))         // day must be MMDD
        assertNull(BuildTag.parse("v0.8.7-rc1"))         // prerelease suffix, not a build
    }

    @Test
    fun ordersByVersionThenDayThenBuild() {
        // same day, build number
        assertTrue(BuildTag.parse("0.8.7-1015-1")!! < BuildTag.parse("0.8.7-1015-4")!!)
        // next day resets the build but is newer
        assertTrue(BuildTag.parse("0.8.7-1015-9")!! < BuildTag.parse("0.8.7-1016-1")!!)
        // a version bump beats an older version whatever the day/build
        assertTrue(BuildTag.parse("0.8.7-1231-9")!! < BuildTag.parse("0.8.8-0101-1")!!)
        // the examples from the spec
        assertEquals(0, BuildTag.parse("0.8.7-1015-1")!!.compareTo(BuildTag.parse("0.8.7-1015-1")!!))
    }

    @Test
    fun formatsDatedTag() {
        assertEquals("0.8.7-1015-1", BuildTag(AppVersion(0, 8, 7), 1015, 1).toString())
    }
}