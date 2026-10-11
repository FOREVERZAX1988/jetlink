package io.zoompilot.jetlink.presence

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
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
        val ours = setOf("192.168.43.1")
        assertEquals(Presence.Here, PresenceMonitor.presence("192.168.43.1", ours))
        // the comma joined another hotspot in the same subnet and dials that
        // one's device, not this phone: not the link this phone is listening for
        assertEquals(Presence.Elsewhere, PresenceMonitor.presence("192.168.10.1", ours))
        // a comma that names no gateway is read as before, as on this hotspot
        assertEquals(Presence.Here, PresenceMonitor.presence(null, ours))
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
