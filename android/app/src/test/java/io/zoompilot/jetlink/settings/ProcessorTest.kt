package io.zoompilot.jetlink.settings

import org.junit.Assert.assertEquals
import org.junit.Test

/** What each kind of phone may choose to run the model on. */
class ProcessorTest {
    @Test
    fun aSnapdragonOffersQnn() {
        assertEquals(listOf(Processor.NpuGpu, Processor.Npu, Processor.Gpu), Processor.choices(qualcomm = true, emulator = false, current = Processor.NpuGpu))
        // the CPU once chosen, so it can be chosen away from
        assertEquals(Processor.entries, Processor.choices(qualcomm = true, emulator = false, current = Processor.Cpu))
    }

    @Test
    fun anyOtherPhoneOffersTheGpuAndCpu() {
        // QNN on a Google Tensor leaves every op to one CPU thread; LiteRT drives its GPU
        assertEquals(listOf(Processor.Gpu, Processor.Cpu), Processor.choices(qualcomm = false, emulator = false, current = Processor.Gpu))
    }

    @Test
    fun theQnnGpuChoiceBecomesLiteRtsGpu() {
        assertEquals(Processor.Gpu, Processor.of("gpu"))
        assertEquals(Processor.Gpu, Processor.of("litert-gpu"))
        assertEquals(Processor.NpuGpu, Processor.of("htp"))
        assertEquals(null, Processor.of("tpu"))
    }

    @Test
    fun theEmulatorOffersEverything() {
        assertEquals(Processor.entries, Processor.choices(qualcomm = false, emulator = true, current = Processor.Cpu))
    }
}
