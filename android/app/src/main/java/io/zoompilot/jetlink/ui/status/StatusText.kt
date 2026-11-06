package io.zoompilot.jetlink.ui.status

import android.content.Context
import io.zoompilot.jetlink.R

/**
 * [StatusL10n] for a caller that cannot compose: the foreground service's
 * notification, which reads the same words out of resources with its own
 * Context. Without this the notification stays English while every screen is
 * translated, which is the one place a Chinese user still sees "Connected".
 */
object StatusText {
    /** The summary with the link it is over: "Connected over USB 3". */
    fun headline(context: Context, state: StatusState): String {
        val summary = state.summary
        val medium = state.medium
        return if ((summary == StatusState.Summary.Connected || summary == StatusState.Summary.ConnectedSlow) && medium != null) {
            context.getString(R.string.summary_connected_over, medium.title)
        } else {
            context.getString(summary.titleRes)
        }
    }

    /** The line under the title: the summary and the model. */
    fun subtitle(context: Context, state: StatusState): String =
        listOfNotNull(headline(context, state), state.modelName).joinToString(" · ")
}
