package io.zoompilot.jetlink

import android.app.Application
import android.content.Context
import io.zoompilot.jetlink.device.DeviceMonitor
import io.zoompilot.jetlink.server.ServerController
import io.zoompilot.jetlink.settings.Settings
import io.zoompilot.jetlink.usb.CommaUsb
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob

class JetlinkApp : Application() {
    lateinit var graph: AppGraph
        private set

    override fun onCreate() {
        super.onCreate()
        graph = AppGraph(this)
    }
}

/** The app's one of everything, for the life of the process. */
class AppGraph(context: Context) {
    val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    val settings = Settings(context)
    val server = ServerController(context, scope)
    val usb = CommaUsb(context)
    val device = DeviceMonitor(context, scope).also { it.start() }
}

val Context.graph: AppGraph get() = (applicationContext as JetlinkApp).graph
