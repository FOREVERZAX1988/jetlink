# Publish a release

Updating an installed server: [updates and rollback](releasing.md). This page:
publishing the app, Python packages, and container images.

A pushed `v*` tag runs the Release workflow: the macOS app, the Python sdist and
wheel, and the container images.

1. Set `__version__` in `jetlink/__init__.py` (`pyproject.toml` reads it), add a
   `Jetlink vX.Y.Z` section at the top of `CHANGELOG.md`, and commit.
   - The tag must match `__version__`; `macos/scripts/check-version.sh` checks
     before the build.
   - The section becomes the release notes: write what installers will notice,
     not how it was done. Without one, GitHub generates the notes.
2. Tag and push (replace `0.3.0`):

```bash
git tag v0.3.0
git push origin v0.3.0
```

3. Watch **Actions > Release**. The macOS job builds, smoke-tests and notarizes
   the app; each image builds on a native runner for its architecture.
4. Check the release page: `Jetlink-0.3.0-macOS.dmg`, `SHA256SUMS`, the sdist,
   the wheel, and notes made of the changelog section, the installer command and
   the GHCR image lines.

Prereleases: a hyphen (`v0.3.0-rc1`) or a PEP 440 suffix (`v0.3.0a1`,
`v0.3.0b2`, `v0.3.0rc1`) publishes as a prerelease. Use the same version in
`jetlink/__init__.py` and the tag, minus the leading `v`. Prefer PEP 440
suffixes to avoid wheel filename normalization.

## Installing the app

Open the DMG and drag Jetlink to Applications. Verify against `SHA256SUMS`:

```bash
shasum -a 256 -c SHA256SUMS
```

## The container images

Each release pushes two images to `ghcr.io/zoompilot/jetlink`, tagged with the
full version and with `major.minor`:

| Tag | Platform | Base |
| --- | --- | --- |
| `0.4.0-cuda` | linux/amd64 (NVIDIA PCs, driver 580+) and linux/arm64 (JetPack 7.2+) | `nvidia/cuda:13.2.1-base-ubuntu24.04` |
| `0.4.0-jetpack6`, also `0.4.0-jetson` | linux/arm64, JetPack 6 | `l4t-jetpack:r36.4.0` |

- `-cuda` is one tag for two architectures; Docker pulls the machine's.
- The JetPack 6 image needs L4T r36 and the NVIDIA container runtime; it does
  not run on JetPack 7 or generic Arm servers.
- No `latest` tag.
- Each push to `main` also publishes `edge-cuda` and `edge-jetpack6` (Docker
  Images workflow), which the installer pulls with `--ref main`. With no image
  for a machine, the installer builds it there.
- A failed image job does not block the release; the notes say which failed,
  and `install.sh --build` still works.

## Signing secrets

Releases are signed with a Developer ID and notarized. A fork without these
secrets gets an ad hoc signed ZIP and no DMG; the workflow step "Report the
signing mode" says which mode ran.

| Secret | What |
| --- | --- |
| `MACOS_CERTIFICATE_P12_BASE64` | the Developer ID Application certificate with its private key, exported from Keychain Access as a .p12 and base64 encoded |
| `MACOS_CERTIFICATE_PASSWORD` | the .p12 password |
| `KEYCHAIN_PASSWORD` | any random string; it locks the temporary keychain the runner builds in |
| `NOTARY_KEY_ID` | the App Store Connect API key id |
| `NOTARY_ISSUER_ID` | the issuer id of that key |
| `NOTARY_PRIVATE_KEY_P8_BASE64` | the key's .p8 file, base64 encoded |

- Notary key: a Team key with the Developer role, made under **Users and
  Access > Integrations** in App Store Connect.
- Encode a certificate with `base64 -i cert.p12 | pbcopy`; for
  `NOTARY_PRIVATE_KEY_P8_BASE64` encode the `.p8` file instead.
