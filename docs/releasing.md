# Updates and rollback

The comma build and the Jetlink server must be compatible. Update both while
parked.

## Updating

1. Update the comma first from **Settings > Software** and let it reboot.
2. Update the server using the method you installed:

   - Jetson or Linux PC with the installer: run `jetlink update`. It keeps your
     settings and restarts Jetlink. If the update fails, it restores the
     previous server.
   - Mac app: quit Jetlink, replace it with the new release, and reopen it.
   - Source install: run `git pull` from the Jetlink checkout. For the Mac
     script, restart `scripts/run-mac.sh`; recreate `.venv` if dependencies
     changed.

3. Plug in while parked and wait for the green icon. A new Jetlink or model may
   need another engine build. Cached engines stay valid across updates that do
   not change the model or runtime.

## Rolling back

Turn off **Settings > Models > Accelerator Link** to stop using Jetlink
immediately. To roll back, restore the previous comma build and the previous
server together; restoring one side can leave them incompatible. Keep the model
cache.

With the installer, run it with the release or commit to go back to:

```bash
curl -fsSL https://raw.githubusercontent.com/zoompilot/jetlink/v0.4.0/install.sh | bash -s -- --ref v0.4.0
```

For a specific version or a manual Docker rollback, see the
[installation reference](installation-reference.md#versions-and-manual-rollback).

<a id="which-jetlink-to-run"></a>

## Maintainer reference

Release workflows, container tags, and signing secrets are in the
[publishing guide](publishing.md).

<a id="installing-the-app"></a>
<a id="the-container-images"></a>
<a id="signing-secrets"></a>
