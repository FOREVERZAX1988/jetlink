package io.zoompilot.jetlink.server

import android.content.Context
import android.net.wifi.WifiManager

/**
 * The Wi-Fi lock held while a comma is served over a link that is not the
 * cable.
 *
 * The CPU wake lock (PowerManager.PARTIAL_WAKE_LOCK, held in ServerService) is
 * a different power domain: a phone can keep its CPU up and still let its Wi-Fi
 * radio sleep between beacons, and every reply then waits for the next one.
 * Measured on the comma side of a Wi-Fi link (2026-10-16, 5180 MHz, RSSI -65):
 * 26.8 ms of ping at a 0.2 s interval against 7.0 ms at 0.05 s, max 118 ms --
 * a whole 20 Hz frame budget spent waking a radio that 20 frames a second would
 * have kept awake anyway.
 *
 * Not held over the cable: the frames go through USB there and this is battery
 * the app should not spend (the app's own engine host backs off to save it).
 * Not held while merely listening for a comma either: the presence handshake
 * is one packet a second, and a phone in a car park should sleep.
 */
class WifiLink(context: Context) {
    private val wifi = context.applicationContext.getSystemService(WifiManager::class.java)
    private var lock: WifiManager.WifiLock? = null

    /** Hold it while a comma is connected over a medium that needs it, drop it when it does not. */
    fun hold(keep: Boolean, medium: String?) {
        val wanted = keep && wanted(medium)
        val held = lock
        if (wanted && held == null) {
            lock = create().apply { setReferenceCounted(false); acquire() }
        } else if (!wanted && held != null) {
            held.release()
            lock = null
        }
    }

    @Suppress("DEPRECATION")  // the fallback below, deprecated in favour of the mode above it
    private fun create(): WifiManager.WifiLock = try {
        // The mode that asks the stack for low latency and no power save, which
        // is what the reply needs (API 29; the phone here reports
        // WIFI_FEATURE_LOW_LATENCY). The older high-performance mode is the
        // fallback for a stack that will not take it: createWifiLock is the
        // call that refuses a mode it does not know, and a phone old enough to
        // refuse knows only the one it deprecates.
        wifi.createWifiLock(WifiManager.WIFI_MODE_FULL_LOW_LATENCY, TAG)
    } catch (e: IllegalArgumentException) {
        wifi.createWifiLock(WifiManager.WIFI_MODE_FULL_HIGH_PERF, TAG)
    }

    companion object {
        /** What the radio's lock log calls it, and the only thing it is held for. */
        private const val TAG = "jetlink:serving"

        /**
         * Is this a medium the lock is worth holding for? Anything but the
         * cable. The names are the snapshot's, from the comma's own hello
         * (JetlinkKit's LinkMedium: usb3, usb2, usb1, usb, tcp, wifi, wifi24);
         * a server too old to name one is not a reason to hold one, and neither
         * is a link that is not connected yet -- `medium` is only sent while
         * one is.
         *
         * An empty name is not a medium either: a lock held on a guess is
         * battery spent on a guess.
         */
        fun wanted(medium: String?): Boolean = !medium.isNullOrEmpty() && !medium.startsWith("usb")
    }
}
