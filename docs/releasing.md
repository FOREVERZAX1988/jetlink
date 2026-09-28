# Updates and rollback

The comma build and Jetlink server must be compatible. Update offroad, with
both devices powered and online.

## Updating

1. Update the comma in **Settings > Software** and let it reboot.
2. Update the server:

   - Jetson or Linux PC (installer): run `jetlink update`. It moves to the
     newest release, keeps your settings, and restores the previous server if
     the update fails.
   - Mac app: quit Jetlink, replace it with the new release, and reopen it.
   - iPhone app: run `git pull` in the checkout, then click **Run** in Xcode.
   - Source install: run `git pull` in the checkout. Mac script: restart
     `scripts/run-mac.sh`; recreate `.venv` if dependencies changed.

3. Connect the comma offroad and wait for green. A new Jetlink or model may
   rebuild the engine.

## Rolling back

* Stop using Jetlink now: set **Settings > Models > Accelerator Link** to **Off**.
* Roll back the comma build and server together; one alone can leave them
  incompatible. Keep the model cache.
* Installer: rerun it with the release or commit to go back to:

```bash
curl -fsSL https://raw.githubusercontent.com/zoompilot/jetlink/v0.4.0/install.sh | bash -s -- --ref v0.4.0
```

It stays there until `jetlink update --ref latest`.

Specific versions and manual Docker rollback:
[installation reference](installation-reference.md#versions-and-manual-rollback).

<a id="which-jetlink-to-run"></a>

## Maintainer reference

Release workflows, container tags, and signing secrets:
[publishing guide](publishing.md).

<a id="installing-the-app"></a>
<a id="the-container-images"></a>
<a id="signing-secrets"></a>
