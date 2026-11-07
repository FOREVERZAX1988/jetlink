package io.zoompilot.jetlink.server

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/** When a Wi-Fi lock is worth holding: a comma's link that is not the cable. */
class WifiLinkTest {
    @Test
    fun theCableIsTheOnlyMediumWithoutOne() {
        for (medium in listOf("usb", "usb3", "usb2", "usb1")) {
            assertFalse(medium, WifiLink.wanted(medium))
        }
    }

    @Test
    fun theRadioMediaGetOne() {
        for (medium in listOf("wifi", "wifi24", "tcp")) {
            assertTrue(medium, WifiLink.wanted(medium))
        }
    }

    @Test
    fun aMediumNobodyNamedGetsNone() {
        assertFalse(WifiLink.wanted(null))
        assertFalse(WifiLink.wanted(""))
    }
}
