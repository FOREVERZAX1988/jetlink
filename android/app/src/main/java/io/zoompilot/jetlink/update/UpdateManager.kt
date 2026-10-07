package io.zoompilot.jetlink.update

import android.content.Context
import android.content.Intent
import android.net.Uri
import androidx.core.content.FileProvider
import io.zoompilot.jetlink.BuildConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext
import java.io.File

/**
 * The app's self-update through the GitHub release: [check] asks the API
 * for the newest release, [download] fetches its APK into the cache and
 * [install] hands it to the system installer.
 *
 * The release key the APKs are signed with lets Android install a newer
 * one over this app (docs/publishing.md), so an update never asks to
 * uninstall first.
 */
class UpdateManager(
    private val context: Context,
    private val github: GithubReleases = GithubReleases(),
    private val currentVersion: AppVersion = AppVersion.parse(BuildConfig.VERSION_NAME) ?: AppVersion(0, 0, 0),
) {
    private val _state = MutableStateFlow<UpdateState>(UpdateState.Unknown)
    val state: StateFlow<UpdateState> = _state.asStateFlow()

    /** The release the latest check found, with its APK asset, when one is newer. */
    @Volatile
    private var pending: Pair<GithubRelease, GithubAsset>? = null

    val cacheDir: File = File(context.cacheDir, "downloads").apply { mkdirs() }

    /** Asks GitHub for the newest release and compares it with the installed version. */
    suspend fun check() {
        _state.value = UpdateState.Checking
        val release = withContext(Dispatchers.IO) {
            try {
                github.latest()
            } catch (e: Exception) {
                null
            }
        }
        if (release == null) {
            _state.value = UpdateState.Error("no release")
            return
        }
        val parsed = AppVersion.parse(release.tag_name)
        if (parsed == null) {
            _state.value = UpdateState.Error("bad tag ${release.tag_name}")
            return
        }
        val asset = github.apk(release)
        if (parsed > currentVersion && asset != null) {
            pending = release to asset
            _state.value = UpdateState.Available(parsed)
        } else {
            _state.value = UpdateState.Current
        }
    }

    /**
     * Downloads the pending update's APK into the cache. Returns the file,
     * or null when nothing is pending or the download failed.
     */
    suspend fun download(onProgress: (Long, Long) -> Unit = { _, _ -> }): File? {
        val (_, asset) = pending ?: return null
        return withContext(Dispatchers.IO) {
            try {
                val file = File(cacheDir, asset.name)
                val bytes = downloadToFile(asset.browser_download_url, file, onProgress = onProgress)
                if (bytes > 0) file else null
            } catch (e: Exception) {
                null
            }
        }
    }

    /** Asks the system to install [apk], through the FileProvider. */
    fun install(apk: File) {
        val uri: Uri = FileProvider.getUriForFile(context, "${context.packageName}.files", apk)
        val intent = Intent(Intent.ACTION_VIEW).apply {
            setDataAndType(uri, "application/vnd.android.package-archive")
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_ACTIVITY_NEW_TASK)
        }
        context.startActivity(intent)
    }
}