package io.zoompilot.jetlink.presence

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** The announcement and the answer to it, as jetlink.comma.wifi.Presence sends and reads them. */
class PresenceMonitorTest {
    @Test
    fun anAnnouncementNamesItsProtocolAndItsId() {
        assertEquals("9b3f0c1d", PresenceMonitor.announceId("""{"jetlink":"presence","v":1,"id":"9b3f0c1d"}"""))
    }

    @Test
    fun theAnswerCarriesThatIdAndTheAck() {
        val answer = Json.parseToJsonElement(PresenceMonitor.answer("9b3f0c1d")).jsonObject
        assertEquals("presence", (answer["jetlink"] as? JsonPrimitive)?.contentOrNull)
        assertEquals("1", (answer["v"] as? JsonPrimitive)?.contentOrNull)
        assertEquals("true", (answer["ack"] as? JsonPrimitive)?.contentOrNull)
        assertEquals("9b3f0c1d", (answer["id"] as? JsonPrimitive)?.contentOrNull)
    }

    @Test
    fun anAnnouncementNamesTheGatewayItsCommaDials() {
        assertEquals(
            "192.168.43.1",
            PresenceMonitor.announceGateway("""{"jetlink":"presence","v":1,"id":"9b3f0c1d","gw":"192.168.43.1"}"""),
        )
        // an older comma says nothing of one: the app reads it as before
        assertNull(PresenceMonitor.announceGateway("""{"jetlink":"presence","v":1,"id":"9b3f0c1d"}"""))
        assertNull("another kind", PresenceMonitor.announceGateway("""{"jetlink":"snapshot","v":1,"gw":"192.168.43.1"}"""))
    }

    @Test
    fun theGatewayIsThisPhoneOnlyWhenItIsOneOfItsOwn() {
        val hotspot = setOf(LocalAddress("192.168.43.1", 24))
        assertEquals(Presence.Here, PresenceMonitor.presence(null, "192.168.43.1", "192.168.43.42", hotspot))
        // the comma joined another network and dials that one's device, not this
        // phone: the announcement comes from a subnet this phone is not on
        assertEquals(Presence.Elsewhere, PresenceMonitor.presence(null, "192.168.10.1", "192.168.10.77", hotspot))
        // a comma that names no gateway is read as before, as on this hotspot
        assertEquals(Presence.Here, PresenceMonitor.presence(null, null, null, hotspot))
    }

    @Test
    fun aCommaOnThisPhonesOwnLanIsOnItsWay() {
        // the router is the gateway and this phone is a client of it: discovery
        // finds the app and dials it, so that is this phone's link too
        val lan = setOf(LocalAddress("192.168.10.50", 24))
        assertEquals(Presence.Here, PresenceMonitor.presence(null, "192.168.10.251", "192.168.10.79", lan))
    }

    @Test
    fun theHostTheCommaNamesIsTheOneItIsReadAgainst() {
        // once an app answers, the comma names the host it dials, and that
        // settles whose link it is - on one LAN a second app would otherwise
        // read every announcement as its own
        val lan = setOf(LocalAddress("192.168.10.50", 24))
        assertEquals(Presence.Here, PresenceMonitor.presence("192.168.10.50", "192.168.10.251", "192.168.10.79", lan))
        assertEquals(Presence.Elsewhere, PresenceMonitor.presence("192.168.10.65", "192.168.10.251", "192.168.10.79", lan))
    }

    @Test
    fun anAnnouncementNamesTheHostItsCommaDials() {
        assertEquals(
            "192.168.10.50",
            PresenceMonitor.announceTarget("""{"jetlink":"presence","v":1,"id":"9b3f0c1d","to":"192.168.10.50"}"""),
        )
        assertNull(PresenceMonitor.announceTarget("""{"jetlink":"presence","v":1,"id":"9b3f0c1d"}"""))
        assertNull("another kind", PresenceMonitor.announceTarget("""{"jetlink":"snapshot","v":1,"to":"192.168.10.50"}"""))
    }

    @Test
    fun aSubnetIsReadOfThePhonesOwnPrefix() {
        val lan = LocalAddress("192.168.10.50", 24)
        assertTrue(lan.sharesSubnet("192.168.10.251"))
        assertFalse(lan.sharesSubnet("192.168.11.50"))
        assertTrue(LocalAddress("10.0.0.44", 16).sharesSubnet("10.0.9.9"))
        assertFalse("a /32 names no LAN", LocalAddress("192.168.10.50", 32).sharesSubnet("192.168.10.50"))
        assertFalse("a /0 names every host there is", LocalAddress("10.0.0.1", 0).sharesSubnet("8.8.8.8"))
        assertFalse(lan.sharesSubnet("not an address"))
    }

    @Test
    fun aPhoneHoldsTheAddressesOfItsOwnInterfaces() {
        // both seams off the real interfaces: the loopback is never a link, and
        // every address comes with the prefix its interface gave it
        assertTrue(localAddresses().none { it.address.startsWith("127.") })
        assertTrue(localAddresses().all { it.prefix in 0..32 })
    }

    @Test
    fun nothingElseIsAnAnnouncement() {
        assertNull("another kind", PresenceMonitor.announceId("""{"jetlink":"snapshot","v":1,"id":"9b3f0c1d"}"""))
        assertNull("another version", PresenceMonitor.announceId("""{"jetlink":"presence","v":2,"id":"9b3f0c1d"}"""))
        assertNull("no id", PresenceMonitor.announceId("""{"jetlink":"presence","v":1}"""))
        assertNull("not json", PresenceMonitor.announceId("hello"))
        assertNull("not an object", PresenceMonitor.announceId("[]"))
    }
}
