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

    /**
     * A comma this phone is the link of: it dialed an address this phone holds
     * (its hotspot), or it is on a subnet of this phone's and dials this one
     * (a router LAN, where discovery finds the app and dials it).
     */
    Here,

    /**
     * A comma answered, but the link it is dialing is another device's: another
     * hotspot, or another app on this network. Not the comma this phone is
     * listening for.
     */
    Elsewhere,
}

/**
 * One of this phone's own IPv4 addresses, and the subnet it is on, so that a
 * comma's address can be read as either the phone itself or a peer it reaches
 * without a router in between.
 */
data class LocalAddress(val address: String, val prefix: Int) {
    /** This phone holds `other`. */
    fun holds(other: String): Boolean = address == other

    /**
     * `other` is a host on this address's subnet: a comma here reaches this
     * phone directly, which is what a router LAN's link is. A prefix that names
     * no hosts, or every host there is, says nothing about a LAN and is not one.
     */
    fun sharesSubnet(other: String): Boolean {
        if (prefix !in 8..30) return false
        val mine = packed(address) ?: return false
        val theirs = packed(other) ?: return false
        val mask = -1 shl (32 - prefix)
        return (mine and mask) == (theirs and mask)
    }

    /** The dotted quad as an Int of its four octets, or null when it is not one. */
    private fun packed(quad: String): Int? {
        val parts = quad.split('.')
        if (parts.size != 4) return null
        var value = 0
        for (part in parts) {
            val octet = part.toIntOrNull() ?: return null
            if (octet !in 0..255) return null
            value = (value shl 8) or octet
        }
        return value
    }
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
 * A comma reaches this phone two ways, and hears it on two, because the phone
 * can be either the gateway of its own hotspot or a client of the router both
 * are on. On the hotspot the comma unicasts to its gateway, which is this
 * phone; on a router it unicasts to the router, which drops it, and also sends
 * the same announcement to the subnet's broadcast, where this phone hears it
 * (jetlink.comma.wifi.announce_targets). Either way the announcement names what
 * the comma dials - `to` once an app has answered that host, `gw` before that -
 * and a gateway on one of this phone's own subnets is the router of a LAN this
 * phone is on, not some other network's.
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
    private val addresses: () -> Set<LocalAddress> = ::localAddresses,
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
                val present = presence(announceTarget(text), announceGateway(text),
                                       packet.address?.hostAddress, addresses())
                mutableState.value = present
                // the answer says the comma dials this phone, so it is the one
                // to answer; one dialing another device is not this link, and an
                // answer would only tell it an app is here
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
         * The host the comma that sent [text] will dial, or null before an app
         * has answered it: a comma names the app that answered - that is the
         * address discovery would dial on this network too - so a second app on
         * one LAN is told whose link the comma is on.
         */
        fun announceTarget(text: String): String? = field(text, "to")

        /**
         * What an announcement means to a phone holding [addresses], sent from
         * [sender]: this phone's link when the comma names an address of this
         * phone's as the one it dials, or, naming none, when its gateway is this
         * phone (the phone's hotspot) or another host on a subnet this phone is
         * on (a router LAN, where the gateway is the router). Everything else
         * answered but dials another device. `sender` is the announcement's own
         * source address, when the socket would say.
         */
        fun presence(to: String?, gateway: String?, sender: String?, addresses: Set<LocalAddress>): Presence = when {
            // the comma named the host it dials: this phone is that link, or it is another app's
            to != null -> if (addresses.any { it.holds(to) }) Presence.Here else Presence.Elsewhere
            // a comma old enough to name no host is read as before, as on this hotspot
            gateway == null -> Presence.Here
            addresses.any { it.holds(gateway) } -> Presence.Here
            // a gateway this phone does not hold, on a subnet it is on: the
            // router a LAN announcement comes through, and a comma that can
            // dial this phone over it
            sender != null && addresses.any { it.sharesSubnet(sender) } -> Presence.Here
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

/**
 * This phone's own IPv4 addresses and their subnets, off the loopback: the
 * addresses a comma dials on this phone's hotspot, and the subnets one on the
 * same router reaches it over. The prefix comes from the interface, so a
 * router's LAN is a subnet this phone is on and not a guess at one.
 */
internal fun localAddresses(): Set<LocalAddress> = runCatching {
    NetworkInterface.getNetworkInterfaces().asSequence()
        .filterNotNull()
        .flatMap { it.interfaceAddresses.asSequence() }
        .mapNotNull { held ->
            val address = held.address as? Inet4Address ?: return@mapNotNull null
            if (address.isLoopbackAddress) return@mapNotNull null
            val host = address.hostAddress?.substringBefore('%') ?: return@mapNotNull null
            LocalAddress(host, held.networkPrefixLength.toInt())
        }
        .toSet()
}.getOrDefault(emptySet())
