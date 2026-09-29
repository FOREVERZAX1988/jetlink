# Updates and rollback

The comma and Jetlink speak one protocol: update them together. Update
offroad, with both devices powered and online.

## Updating

1. Update the comma in **Settings > Software** and let it reboot.
2. Update the server:

   - Jetson or Linux PC (installer): run `jetlink update`. It moves to the
     newest release, keeps your settings, and restores the previous server if
     the update fails. An install from 0.6.0 or earlier also moves out of
     Docker, keeping its models and prepared engines.
   - Mac app: quit Jetlink, replace it with the new release, and reopen it.
   - iPhone app: run `git pull` in the checkout, then click **Run** in Xcode.
   - Android app: run `git pull` in the checkout, then build and install it
     again ([Android development](../android/README.md#build)).
   - Mac terminal: run `git pull`, then build `jetlink-server` again
     ([from a terminal](platforms.md#from-a-terminal)).

3. Connect the comma offroad and wait for green. A new Jetlink or model may
   prepare the engine again.

If only one side was updated, the comma drives on its small model, and the
server's log (`jetlink logs`) or the comma's says which side is behind:

| The log says | Update |
| --- | --- |
| `update the comma's jetlink package` | the comma, in **Settings > Software** |
| `update jetlink on the Jetson` | Jetlink on the server |

## Rolling back

* Stop using Jetlink now: set **Settings > Models > Accelerator Link** to **Off**.
* Roll back the comma build and server together; one alone can leave them
  incompatible. Keep the model cache.
* Installer: `jetlink update --ref v0.7.0` (replace with the release to go
  back to). It stays there until `jetlink update --ref latest`.
* Going back to 0.6.0 puts the Docker server back. To come forward from it,
  use the installer, since 0.6.0's `jetlink update` cannot:

```bash
curl -fsSL https://raw.githubusercontent.com/zoompilot/jetlink/main/install.sh | bash -s -- --update --ref latest
```

Manual rollback: [installation reference](installation-reference.md#versions-and-manual-rollback).

<a id="which-jetlink-to-run"></a>

## Maintainer reference

Releasing: [publishing guide](publishing.md). A version bump in
`jetlink/__init__.py` means running `JetlinkKit/Scripts/make_pins.py` again;
each release attaches the Linux server tarballs the installer downloads.

<a id="installing-the-app"></a>
<a id="the-container-images"></a>
<a id="signing-secrets"></a>
