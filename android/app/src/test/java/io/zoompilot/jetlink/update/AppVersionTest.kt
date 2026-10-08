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
}