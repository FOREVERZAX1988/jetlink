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
    fun nothingElseIsAnAnnouncement() {
        assertNull("another kind", PresenceMonitor.announceId("""{"jetlink":"snapshot","v":1,"id":"9b3f0c1d"}"""))
        assertNull("another version", PresenceMonitor.announceId("""{"jetlink":"presence","v":2,"id":"9b3f0c1d"}"""))
        assertNull("no id", PresenceMonitor.announceId("""{"jetlink":"presence","v":1}"""))
        assertNull("not json", PresenceMonitor.announceId("hello"))
        assertNull("not an object", PresenceMonitor.announceId("[]"))
    }
}
