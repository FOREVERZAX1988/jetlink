package io.zoompilot.jetlink.settings

import android.content.Context
import android.content.SharedPreferences
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/** Where the model runs, as the server names the device: an OrtProfile, or LiteRT's. */
enum class Processor(val id: String, val title: String) {
    /** The vision trunk on the NPU, the rest on the GPU: the Mac's split. QNN, a Snapdragon's. */
    NpuGpu("htp", "NPU + GPU"),

    /** The whole model on the NPU, prepared as the iPhone's. QNN, a Snapdragon's. */
    Npu("htp-whole", "NPU"),

    /** The whole model on the GPU through LiteRT, which drives any phone's: Adreno, Mali, PowerVR. */
    Gpu("litert-gpu", "GPU"),

    /** The CPU: the emulator and tests. Seconds a frame with a real model. */
    Cpu("cpu", "CPU");

    /** Runs on QNN, which needs a Snapdragon. */
    val usesQnn: Boolean get() = this == NpuGpu || this == Npu

    companion object {
        /** `gpu` was QNN on the Adreno, before LiteRT drove every phone's GPU. */
        fun of(id: String?): Processor? = if (id == "gpu") Gpu else entries.firstOrNull { it.id == id }

        /**
         * What a phone can choose. The NPU choices run QNN, which needs a
         * Snapdragon: anywhere else it leaves every op to one CPU thread,
         * minutes a frame. The GPU is LiteRT's, on any phone. The CPU is
         * offered on a Snapdragon only once chosen; the emulator offers
         * everything, for testing.
         */
        fun choices(qualcomm: Boolean, emulator: Boolean, current: Processor): List<Processor> = when {
            emulator -> entries
            qualcomm -> entries.filter { it != Cpu || current == Cpu }
            else -> listOf(Gpu, Cpu)
        }
    }
}

/** The few things worth changing on a phone, kept in SharedPreferences. */
data class SettingsValues(
    /** Where bench tools such as `bench_link.py --host` reach the phone. */
    val port: Int = 5599,
    val processor: Processor = Processor.NpuGpu,
    /** The NPU held in burst mode between frames rather than let it settle. */
    val keepNpuAwake: Boolean = true,
    /** A CPU core kept busy between frames. */
    val keepCpuAwake: Boolean = false,
    /** The screen stays on while Jetlink is on screen. */
    val keepScreenOn: Boolean = true,
)

class Settings(context: Context) {
    private val prefs: SharedPreferences = context.getSharedPreferences("settings", Context.MODE_PRIVATE)
    private val state = MutableStateFlow(read())
    val values: StateFlow<SettingsValues> = state.asStateFlow()

    fun update(change: (SettingsValues) -> SettingsValues) {
        val next = change(state.value)
        prefs.edit()
            .putInt(PORT, next.port)
            .putString(PROCESSOR, next.processor.id)
            .putBoolean(KEEP_NPU_AWAKE, next.keepNpuAwake)
            .putBoolean(KEEP_CPU_AWAKE, next.keepCpuAwake)
            .putBoolean(KEEP_SCREEN_ON, next.keepScreenOn)
            .apply()
        state.value = next
    }

    private fun read(): SettingsValues {
        val defaults = SettingsValues(processor = defaultProcessor())
        val port = prefs.getInt(PORT, defaults.port)
        return SettingsValues(
            port = if (port in 1..65535) port else defaults.port,
            // a QNN choice from before a phone without a Snapdragon was told apart
            processor = Processor.of(prefs.getString(PROCESSOR, null))
                ?.takeIf { it in Processor.choices(Chip.isQualcomm, Chip.isEmulator, it) }
                ?: defaults.processor,
            keepNpuAwake = prefs.getBoolean(KEEP_NPU_AWAKE, defaults.keepNpuAwake),
            keepCpuAwake = prefs.getBoolean(KEEP_CPU_AWAKE, defaults.keepCpuAwake),
            keepScreenOn = prefs.getBoolean(KEEP_SCREEN_ON, defaults.keepScreenOn),
        )
    }

    private companion object {
        const val PORT = "port"
        const val PROCESSOR = "processor"
        const val KEEP_NPU_AWAKE = "keepNpuAwake"
        const val KEEP_CPU_AWAKE = "keepCpuAwake"
        const val KEEP_SCREEN_ON = "keepScreenOn"

        /** The split on a Snapdragon, the GPU on any other phone, the CPU on the emulator. */
        fun defaultProcessor(): Processor = when {
            Chip.isEmulator -> Processor.Cpu
            Chip.isQualcomm -> Processor.NpuGpu
            else -> Processor.Gpu
        }
    }
}
