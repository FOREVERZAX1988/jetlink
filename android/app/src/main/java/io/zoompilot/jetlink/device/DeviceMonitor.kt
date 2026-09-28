package io.zoompilot.jetlink.device

import android.app.ActivityManager
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.PowerManager
import androidx.core.content.ContextCompat
import io.zoompilot.jetlink.server.Native
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

/** The phone's own state, which is what slows a model down first. */
data class DeviceHealth(
    /** "nominal", "fair", "serious" or "critical", as the benchmark names them. */
    val thermal: String = "nominal",
    /** Android's thermal headroom: 1.0 is where it starts throttling. Null when unknown. */
    val headroom: Float? = null,
    /** The battery's temperature, °C. */
    val batteryTemp: Float? = null,
    val batteryLevel: Int? = null,
    val charging: Boolean = false,
    val powerSave: Boolean = false,
    val availableMemory: Long = 0,
    val lowMemory: Boolean = false,
) {
    /** Under 1 GB free a model may not fit to prepare, as on the iPhone. */
    val memoryTight: Boolean get() = availableMemory in 1 until 1_000_000_000L || lowMemory
}

/**
 * Battery, heat and memory, polled once a second while the app runs, and
 * the thermal state told to the server for its benchmark reports.
 */
class DeviceMonitor(private val context: Context, private val scope: CoroutineScope) {
    private val power = context.getSystemService(PowerManager::class.java)
    private val activity = context.getSystemService(ActivityManager::class.java)
    private val state = MutableStateFlow(DeviceHealth())
    val health: StateFlow<DeviceHealth> = state.asStateFlow()

    private var battery: Intent? = null
    /** What the server was last told, so it hears the first state and every change. */
    private var reported: String? = null

    fun start() {
        val receiver = object : BroadcastReceiver() {
            override fun onReceive(context: Context, intent: Intent) {
                battery = intent
            }
        }
        battery = ContextCompat.registerReceiver(
            context, receiver, IntentFilter(Intent.ACTION_BATTERY_CHANGED), ContextCompat.RECEIVER_NOT_EXPORTED,
        )
        power.addThermalStatusListener(context.mainExecutor) { refresh() }
        scope.launch(Dispatchers.Default) {
            while (isActive) {
                refresh()
                delay(1000)
            }
        }
    }

    private fun refresh() {
        val memory = ActivityManager.MemoryInfo().also(activity::getMemoryInfo)
        val intent = battery
        val level = intent?.let {
            val raw = it.getIntExtra(BatteryManager.EXTRA_LEVEL, -1)
            val scale = it.getIntExtra(BatteryManager.EXTRA_SCALE, 100)
            if (raw >= 0 && scale > 0) raw * 100 / scale else null
        }
        val plugged = (intent?.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0) ?: 0) != 0
        val tenths = intent?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, Int.MIN_VALUE) ?: Int.MIN_VALUE
        val thermal = thermalLabel(power.currentThermalStatus)
        // The headroom forecast is rate limited to about once a second.
        val headroom = power.getThermalHeadroom(10).takeUnless { it.isNaN() }
        val next = DeviceHealth(
            thermal = thermal,
            headroom = headroom,
            batteryTemp = if (tenths != Int.MIN_VALUE) tenths / 10f else null,
            batteryLevel = level,
            charging = plugged,
            powerSave = power.isPowerSaveMode,
            availableMemory = memory.availMem,
            lowMemory = memory.lowMemory,
        )
        if (next.thermal != reported) {
            reported = next.thermal
            Native.reportThermal(next.thermal)
        }
        state.value = next
    }

    companion object {
        /** PowerManager's thermal status in the benchmark's four words. */
        fun thermalLabel(status: Int): String = when (status) {
            PowerManager.THERMAL_STATUS_NONE, PowerManager.THERMAL_STATUS_LIGHT -> "nominal"
            PowerManager.THERMAL_STATUS_MODERATE -> "fair"
            PowerManager.THERMAL_STATUS_SEVERE -> "serious"
            else -> "critical"
        }
    }
}
