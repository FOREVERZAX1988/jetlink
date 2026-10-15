package io.zoompilot.jetlink

import android.app.Application
import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.content.pm.PackageManager
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import io.zoompilot.jetlink.device.DeviceMonitor
import io.zoompilot.jetlink.server.RunState
import io.zoompilot.jetlink.server.ServerController
import io.zoompilot.jetlink.server.ServerService
import io.zoompilot.jetlink.settings.Settings
import io.zoompilot.jetlink.update.UpdateManager
import io.zoompilot.jetlink.update.UpdateState
import io.zoompilot.jetlink.usb.CommaUsb
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch

class JetlinkApp : Application() {
    lateinit var graph: AppGraph
        private set

    override fun onCreate() {
        super.onCreate()
        graph = AppGraph(this)
        graph.checkForUpdatesAtLaunch()
    }
}

/** The app's one of everything, for the life of the process. */
class AppGraph(private val context: Context) {
    val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    val settings = Settings(context)
    val server = ServerController(context, scope)
    val usb = CommaUsb(context)
    val device = DeviceMonitor(context, scope)
    val updates = UpdateManager(context)

    /** The notification channel update alerts come on. */
    val updateChannelId: String = "updates"

    init {
        createUpdateChannel()
    }

    /**
     * The launch check: looks for a newer release in the background and, when
     * [settings.autoUpdate] is on and one is found, raises an update
     * notification. A silent failure stays silent.
     */
    fun checkForUpdatesAtLaunch() {
        if (!settings.values.value.autoUpdate) return
        scope.launch(Dispatchers.IO) {
            updates.check()
            val state = updates.state.value
            if (state is UpdateState.Available) {
                notifyUpdate(state.tag)
            }
        }
    }

    private fun createUpdateChannel() {
        val channel = NotificationChannel(
            updateChannelId,
            context.getString(R.string.settings_updates),
            NotificationManager.IMPORTANCE_DEFAULT,
        )
        context.getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
    }

    /** An update alert: the title and the release tag, tapping it opens the app. */
    private fun notifyUpdate(tag: String) {
        if (ContextCompat.checkSelfPermission(context, android.Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) return
        val pending = android.app.PendingIntent.getActivity(
            context, 0,
            context.packageManager.getLaunchIntentForPackage(context.packageName),
            android.app.PendingIntent.FLAG_IMMUTABLE or android.app.PendingIntent.FLAG_UPDATE_CURRENT,
        )
        val notification = NotificationCompat.Builder(context, updateChannelId)
            .setSmallIcon(android.R.drawable.stat_sys_download_done)
            .setContentTitle(context.getString(R.string.update_notification_title))
            .setContentText(context.getString(R.string.update_notification_text, tag))
            .setContentIntent(pending)
            .setAutoCancel(true)
            .build()
        context.getSystemService(NotificationManager::class.java).notify(1, notification)
    }

    /**
     * Starts the server again with the current settings and hands it the
     * comma if one is plugged in; a stopped service starts, and starts it.
     */
    fun restartServer(context: Context) {
        if (server.runState.value == RunState.Stopped) {
            ServerService.start(context)
            return
        }
        scope.launch(Dispatchers.IO) {
            server.start(settings.values.value)
            usb.connect()
        }
    }
}

val Context.graph: AppGraph get() = (applicationContext as JetlinkApp).graph
