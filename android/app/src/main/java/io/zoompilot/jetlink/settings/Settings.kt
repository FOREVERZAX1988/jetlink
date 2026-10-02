package io.zoompilot.jetlink.settings

import android.content.Context
import android.content.SharedPreferences
import io.zoompilot.jetlink.AppLocale
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/** The server's backends, as its start config names them. */
enum class Backend(val id: String) {
    Ort("ort"),
    LiteRt("litert"),
}

/**
 * Where the model runs: a backend of the server's and its device, an
 * OrtProfile or a LiteRtProfile. The device is also what Settings stores,
 * and no two choices share one.
 */
enum class Processor(val backend: Backend, val device: String, val title: String) {
    /** The vision trunk on the NPU, the rest on the GPU: the Mac's split. QNN, a Snapdragon's. */
    NpuGpu(Backend.Ort, "htp", "NPU + GPU"),

    /** The whole model on the NPU, prepared as the iPhone's. QNN, a Snapdragon's. */
    Npu(Backend.Ort, "htp-whole", "NPU"),

    /** The whole model on the GPU through LiteRT, which drives any phone's: Adreno, Mali, PowerVR. */
    Gpu(Backend.LiteRt, "gpu", "GPU"),

    /** The CPU: the emulator and tests. Seconds a frame with a real model. */
    Cpu(Backend.Ort, "cpu", "CPU");

    /** Runs on QNN, which needs a Snapdragon: onnxruntime anywhere but its CPU. */
    val usesQnn: Boolean get() = backend == Backend.Ort && device != Cpu.device

    companion object {
        /**
         * A stored choice. `gpu` was QNN on the Adreno before LiteRT drove
         * every phone's GPU, and `litert-gpu` LiteRT's before the backend was
         * a setting of its own: both are the GPU now.
         */
        fun of(device: String?): Processor? = if (device == "litert-gpu") Gpu else entries.firstOrNull { it.device == device }

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

/** The app's colour scheme; System follows the phone (the default). */
enum class AppTheme(val id: String) {
    System("system"),
    Light("light"),
    Dark("dark");

    companion object {
        fun of(id: String?): AppTheme? = entries.firstOrNull { it.id == id }
    }
}

/** The mirror a fresh install starts with; more can be added in the settings. */
const val DEFAULT_MIRROR = "https://hf-mirror.com"

/** The few things worth changing on a phone, kept in SharedPreferences. */
data class SettingsValues(
    /** Where bench tools such as `bench_link.py --host` reach the phone. */
    val port: Int = 5599,
    val processor: Processor = Processor.Gpu,
    /** The NPU held in burst mode between frames rather than let it settle. */
    val keepNpuAwake: Boolean = true,
    /** A CPU core kept busy between frames. */
    val keepCpuAwake: Boolean = false,
    /** The screen stays on while Jetlink is on screen. */
    val keepScreenOn: Boolean = true,
    /** The UI language; System follows the phone (the default). */
    val language: AppLocale = AppLocale.System,
    /** The colour scheme; System follows the phone (the default). */
    val theme: AppTheme = AppTheme.System,
    /** Mirror bases the model catalog and downloads try first; the original hosts come last. */
    val mirrors: List<String> = listOf(DEFAULT_MIRROR),
)

class Settings(context: Context) {
    private val prefs: SharedPreferences = context.getSharedPreferences("settings", Context.MODE_PRIVATE)
    private val state = MutableStateFlow(read())
    val values: StateFlow<SettingsValues> = state.asStateFlow()

    fun update(change: (SettingsValues) -> SettingsValues) {
        val next = change(state.value)
        prefs.edit()
            .putInt(PORT, next.port)
            .putString(PROCESSOR, next.processor.device)
            .putBoolean(KEEP_NPU_AWAKE, next.keepNpuAwake)
            .putBoolean(KEEP_CPU_AWAKE, next.keepCpuAwake)
            .putBoolean(KEEP_SCREEN_ON, next.keepScreenOn)
            .putString(LANGUAGE, next.language.id)
            .putString(THEME, next.theme.id)
            .putString(MIRRORS, next.mirrors.joinToString("\n"))
            .apply()
        state.value = next
    }

    /** The stored mirror list, or the default one before the user has touched it. */
    private fun readMirrors(): List<String> {
        if (!prefs.contains(MIRRORS)) return listOf(DEFAULT_MIRROR)
        val stored = prefs.getString(MIRRORS, "").orEmpty().split("\n")
            .map { it.trim() }
            .filter { it.startsWith("https://") || it.startsWith("http://") }
        return stored.distinct()
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
            language = AppLocale.of(prefs.getString(LANGUAGE, null)) ?: defaults.language,
            theme = AppTheme.of(prefs.getString(THEME, null)) ?: defaults.theme,
            mirrors = readMirrors(),
        )
    }

    private companion object {
        const val PORT = "port"
        const val PROCESSOR = "processor"
        const val KEEP_NPU_AWAKE = "keepNpuAwake"
        const val KEEP_CPU_AWAKE = "keepCpuAwake"
        const val KEEP_SCREEN_ON = "keepScreenOn"
        const val LANGUAGE = "language"
        const val THEME = "theme"
        const val MIRRORS = "mirrors"

        /**
         * The GPU on every phone, a Snapdragon's too; the CPU on the emulator.
         * LiteRT's GPU path is the one whose outputs are shown to match in
         * float16 (after the LayerNorm rewrite). QNN's NPU choices stay on a
         * Snapdragon but have run on no phone, and 20 of Cinque Terre V3's 44
         * vision LayerNorms overflow float16 when computed step by step.
         */
        fun defaultProcessor(): Processor = if (Chip.isEmulator) Processor.Cpu else Processor.Gpu
    }
}
