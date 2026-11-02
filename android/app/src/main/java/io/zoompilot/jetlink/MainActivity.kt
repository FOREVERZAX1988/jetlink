package io.zoompilot.jetlink

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.SideEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.platform.LocalView
import androidx.core.view.WindowCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import androidx.core.content.ContextCompat
import io.zoompilot.jetlink.server.ServerService
import io.zoompilot.jetlink.settings.AppTheme
import io.zoompilot.jetlink.ui.JetlinkTheme
import io.zoompilot.jetlink.ui.RootScreen
import kotlinx.coroutines.launch

/**
 * The dashboard. Opening it starts the server, which then runs in its
 * foreground service until stopped from the notification.
 *
 * Launch extras, for benches and screenshots as on the iPhone:
 * `adb shell am start -n io.zoompilot.jetlink.android/io.zoompilot.jetlink.MainActivity -e tab models`
 * opens a tab (status, models, benchmark, settings, logs, connect), and
 * `--ei benchmark 60` runs a benchmark of that many seconds.
 */
class MainActivity : ComponentActivity() {
    private val notifications = registerForActivityResult(ActivityResultContracts.RequestPermission()) {}

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        ServerService.start(this)
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
            notifications.launch(Manifest.permission.POST_NOTIFICATIONS)
        }
        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                graph.settings.values.collect { values ->
                    if (values.keepScreenOn) {
                        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                    } else {
                        window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                    }
                }
            }
        }
        val tab = intent.getStringExtra("tab")
        val benchmark = intent.getIntExtra("benchmark", 0).takeIf { it > 0 }
        setContent {
            // A manual language and theme override serve everything below, and
            // follow the saved Settings values (Settings.kt), so changing them
            // in the settings screen repaints the whole app in place.
            val values by remember { graph.settings.values }.collectAsStateWithLifecycle()
            val localized = remember(values.language) { L10n.overlay(this, values.language) }
            val dark = when (values.theme) {
                AppTheme.System -> isSystemInDarkTheme()
                AppTheme.Light -> false
                AppTheme.Dark -> true
            }
            // The overlay only changes where resources (stringResource) come
            // from; graph (server, settings) is process-wide and unchanged.
            val view = LocalView.current
            SideEffect {
                WindowCompat.getInsetsController(window, view).isAppearanceLightStatusBars = !dark
            }
            CompositionLocalProvider(
                LocalAppLocale provides values.language,
                LocalOverlayContext provides localized,
            ) {
                JetlinkTheme(dark = dark) {
                    RootScreen(graph = graph, initialTab = tab, launchBenchmark = benchmark)
                }
            }
        }
    }
}
