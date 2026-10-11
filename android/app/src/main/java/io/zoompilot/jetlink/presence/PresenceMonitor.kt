package io.zoompilot.jetlink.presence

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.put
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.Inet4Address
import java.net.InetSocketAddress
import java.net.NetworkInterface
import java.net.SocketAddress
import java.net.SocketTimeoutException

/** What a Wi-Fi presence announcement says of a comma, from this phone. */
enum class Presence {
    /** No comma answered: the Wi-Fi Link setting alone is not a link. */
    None,

    /** A comma on this phone's hotspot: it dialed an address this phone holds. */
    Here,

    /**
     * A comma answered, but it dials another gateway than this phone: it joined
     * a different hotspot, and is not the comma this phone is listening for.
     */
    Elsewhere,
}

/**
 * The phone's end of the Wi-Fi presence handshake (docs/link-protocol.md): a
 * comma whose Jetlink setting is Wi-Fi announces itself once a second to its
 * gateway, this phone, and this answers it.
 *
 * What the Status tab calls connecting over Wi-Fi is a comma answering, never
 * the Wi-Fi Link setting alone. The setting says only that this phone is
 * listening: a comma joined to another network - the phone's home Wi-Fi, a
 * comma that is simply off - is not on its way, and reading the setting as one
 * that is was what made the Wi-Fi link look connected with nothing on it.
 *
 * An announcement names the gateway its comma dials, and this phone answers -
 * and reads the comma as [Presence.Here] - only when that address is one of
 * its own: the two are on one hotspot when the comma dials this phone. A comma
 * reporting another gateway joined a different hotspot ([Presence.Elsewhere]),
 * and saying so beats the "connecting" a bare answer would give.
 */
class PresenceMonitor(
    private val scope: CoroutineScope,
    /** This phone's own IPv4 addresses; a seam so the handshake's decision is testable. */
    private val addresses: () -> Set<String> = ::localAddresses,
) {
    private val mutableState = MutableStateFlow(Presence.None)

    /** A comma on this phone's hotspot, answered within the hold, or why not. */
    val state: StateFlow<Presence> = mutableState.asStateFlow()

    /** Whether the listener should be up. Closed sockets do not survive a stop. */
    @Volatile private var running = false
    @Volatile private var socket: DatagramSocket? = null
    private var job: Job? = null

    /** Listening, once: the Wi-Fi Link setting is on. */
    @Synchronized
    fun start() {
        if (running) return
        running = true
        job = scope.launch(Dispatchers.IO) { listen() }
    }

    /** Not listening, and the answer with it. */
    @Synchronized
    fun stop() {
        running = false
        // a launch that has not started yet never opens its socket, and one
        // that has ends on the close: a receive waiting on a closed socket
        // throws, so the port goes back at once rather than when the hold
        // expires
        job?.cancel()
        job = null
        socket?.close()
        socket = null
        mutableState.value = Presence.None
    }

    private fun listen() {
        val bound = runCatching {
            DatagramSocket(null as SocketAddress?).apply {
                reuseAddress = true
                soTimeout = HOLD_MS
                bind(InetSocketAddress(PORT))
            }
        }.getOrNull() ?: return
        socket = bound
        try {
            val buffer = ByteArray(BUFFER)
            while (running) {
                val packet = DatagramPacket(buffer, buffer.size)
                val text = try {
                    bound.receive(packet)
                    String(packet.data, packet.offset, packet.length, Charsets.UTF_8)
                } catch (e: SocketTimeoutException) {
                    // nothing announced inside the hold: a comma that left the
                    // hotspot, or stopped
                    mutableState.value = Presence.None
                    continue
                } catch (e: Exception) {
                    // stop() closed the socket under us, or it failed: either
                    // way there is nothing to listen on
                    break
                }
                val id = announceId(text) ?: continue
                val present = presence(announceGateway(text), addresses())
                mutableState.value = present
                // the answer says the comma dialed this phone, so it is the one
                // to answer; one that dialed another gateway is not this link,
                // and an answer would only tell it an app is here
                if (present != Presence.Here) continue
                val answer = answer(id).toByteArray(Charsets.UTF_8)
                runCatching { bound.send(DatagramPacket(answer, answer.size, packet.address, packet.port)) }
            }
        } finally {
            bound.close()
            if (socket === bound) socket = null
            mutableState.value = Presence.None
        }
    }

    companion object {
        /** Where the comma announces itself: PRESENCE_PORT in jetlink.comma.wifi. */
        const val PORT = 5600

        /** The version both ends speak; an announcement of another one is not answered. */
        const val VERSION = 1

        private const val KIND = "presence"

        /** How long one announcement counts for. The comma announces every second. */
        private const val HOLD_MS = 5_000

        /** An announcement and the answer to it, well inside one datagram. */
        private const val BUFFER = 512

        /**
         * The id of the announcement [text] is, or null when it is not one this
         * app answers: another kind, another version, or no announcement at all.
         */
        fun announceId(text: String): String? = field(text, "id")

        /**
         * The gateway the comma that sent [text] dials, or null when the
         * announcement says nothing of one: an older comma, whose app reads it
         * as on its way without checking, as it did before this field.
         */
        fun announceGateway(text: String): String? = field(text, "gw")

        /**
         * What an announcement of the gateway [gateway] means to a phone holding
         * [addresses]: on its hotspot when the two are the same, on another one
         * when not. An announcement that names no gateway is read as before, as
         * one on this hotspot.
         */
        fun presence(gateway: String?, addresses: Set<String>): Presence = when {
            gateway == null -> Presence.Here
            gateway in addresses -> Presence.Here
            else -> Presence.Elsewhere
        }

        /** The answer to announcement [id]: the comma reads it as this app listening. */
        fun answer(id: String): String = buildJsonObject {
            put("jetlink", KIND)
            put("v", VERSION)
            put("ack", true)
            put("id", id)
        }.toString()

        /** One field of a presence announcement, or null when [text] is not one. */
        private fun field(text: String, name: String): String? {
            val message = runCatching { Json.parseToJsonElement(text) as? JsonObject }.getOrNull() ?: return null
            if ((message["jetlink"] as? JsonPrimitive)?.contentOrNull != KIND) return null
            if ((message["v"] as? JsonPrimitive)?.content != VERSION.toString()) return null
            return (message[name] as? JsonPrimitive)?.contentOrNull
        }
    }
}

/** This phone's own IPv4 addresses, off the loopback: the addresses a comma on its hotspot dials. */
internal fun localAddresses(): Set<String> = runCatching {
    NetworkInterface.getNetworkInterfaces().asSequence()
        .filterNotNull()
        .flatMap { it.inetAddresses.asSequence() }
        .filterIsInstance<Inet4Address>()
        .filter { !it.isLoopbackAddress }
        .mapNotNull { it.hostAddress?.substringBefore('%') }
        .toSet()
}.getOrDefault(emptySet())
