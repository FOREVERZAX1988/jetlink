package io.zoompilot.jetlink.settings

import org.junit.Assert.assertEquals
import org.junit.Test

/** What each kind of phone may choose to run the model on. */
class ProcessorTest {
    @Test
    fun aSnapdragonOffersQnn() {
        assertEquals(
            listOf(Processor.Auto, Processor.NpuGpu, Processor.Npu, Processor.Gpu),
            Processor.choices(qualcomm = true, tensorNpu = false, emulator = false, current = Processor.NpuGpu),
        )
        // the CPU once chosen, so it can be chosen away from
        assertEquals(
            listOf(Processor.Auto, Processor.NpuGpu, Processor.Npu, Processor.Gpu, Processor.Cpu),
            Processor.choices(qualcomm = true, tensorNpu = false, emulator = false, current = Processor.Cpu),
        )
    }

    @Test
    fun aGoogleTensorOffersItsNpu() {
        assertEquals(
            listOf(Processor.Auto, Processor.TensorNpu, Processor.Gpu, Processor.Cpu),
            Processor.choices(qualcomm = false, tensorNpu = true, emulator = false, current = Processor.Auto),
        )
    }

    @Test
    fun anyOtherPhoneOffersTheGpuAndCpu() {
        // QNN on a Google Tensor leaves every op to one CPU thread; LiteRT drives its GPU
        assertEquals(
            listOf(Processor.Auto, Processor.Gpu, Processor.Cpu),
            Processor.choices(qualcomm = false, tensorNpu = false, emulator = false, current = Processor.Gpu),
        )
    }

    @Test
    fun automaticPicksATensorsNpuAndElseTheGpu() {
        assertEquals(Processor.TensorNpu, Processor.automatic(tensorNpu = true, emulator = false))
        // a Snapdragon too, until QNN's NPU outputs are checked on a phone
        assertEquals(Processor.Gpu, Processor.automatic(tensorNpu = false, emulator = false))
        assertEquals(Processor.Cpu, Processor.automatic(tensorNpu = false, emulator = true))
        assertEquals(Processor.TensorNpu, Processor.Auto.resolved(Processor.TensorNpu))
        assertEquals(Processor.Gpu, Processor.Gpu.resolved(Processor.TensorNpu))
    }

    @Test
    fun theOldDefaultBecomesAutomatic() {
        // every phone's default before Automatic; chosen choices stay
        assertEquals(Processor.Auto, Processor.migrated(Processor.Gpu))
        assertEquals(Processor.NpuGpu, Processor.migrated(Processor.NpuGpu))
        assertEquals(Processor.Cpu, Processor.migrated(Processor.Cpu))
        assertEquals(null, Processor.migrated(null))
    }

    @Test
    fun theQnnGpuChoiceBecomesLiteRtsGpu() {
        assertEquals(Processor.Gpu, Processor.of("gpu"))
        assertEquals(Processor.Gpu, Processor.of("litert-gpu"))
        assertEquals(Processor.NpuGpu, Processor.of("htp"))
        assertEquals(Processor.TensorNpu, Processor.of("npu"))
        assertEquals(Processor.Auto, Processor.of("auto"))
        assertEquals(null, Processor.of("tpu"))
    }

    @Test
    fun onlyOnnxruntimesNpuAndGpuChoicesNeedQnn() {
        assertEquals(listOf(Processor.NpuGpu, Processor.Npu), Processor.entries.filter { it.usesQnn })
        assertEquals(
            listOf(Backend.Ort, Backend.Ort, Backend.LiteRt, Backend.LiteRt, Backend.Ort),
            Processor.entries.filter { it != Processor.Auto }.map { it.backend },
        )
    }

    @Test
    fun theEmulatorOffersEverything() {
        assertEquals(Processor.entries, Processor.choices(qualcomm = false, tensorNpu = false, emulator = true, current = Processor.Cpu))
    }

    @Test
    fun aTensorsGenerationComesFromItsSocModel() {
        assertEquals(5, Chip.tensorGeneration("Tensor G5"))
        assertEquals(3, Chip.tensorGeneration("Tensor G3"))
        assertEquals(null, Chip.tensorGeneration("SM8650"))
        assertEquals(null, Chip.tensorGeneration("Tensor"))
    }
}
