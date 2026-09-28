# Model command reference

Reference for `jetlink-models`. To choose or prepare a model before a drive,
start with [model management](models.md).

## Model identifiers and storage

- A **ref** is a 40-character commit hash from comma's openpilot repository. It
  names a model in sunnypilot's big-model catalog, the list under
  **Settings > Models > Big Model** on the comma.
- A **SHA-256** is the 64-character hash of the ONNX file. One ref resolves to
  exactly one SHA-256, and that never changes.
- `REF_OR_SHA256` takes a 40-character hexadecimal ref or a 64-character
  hexadecimal SHA-256. Anything else is an error.
- The catalog updates on its own, so new models appear without a Jetlink update.
- From Cinque Terre V3 on, the ONNX comes from comma's model repo on Hugging
  Face (`commaai/openpilot_driving_models`).

| Path | What is in it |
| --- | --- |
| `<cache>/models/` | The downloaded ONNX files, about 766 MB each |
| `<cache>/engines/` | The prepared engines and their sidecar files, one per backend and device |
| `<cache>/registry/` | The cached catalog, the resolved pointers, and records of models you imported |

The cache is `JETLINK_CACHE` if set, otherwise `/mnt/data/jetlink` on a Jetson,
`~/Library/Caches/jetlink` on a Mac terminal, and `~/.cache/jetlink` elsewhere.
`--cache DIR` overrides it on every command.

## Commands

```
jetlink-models list      [--refresh] [--json] [--cache DIR]
jetlink-models resolve   REF [--json] [--cache DIR]
jetlink-models fetch     REF_OR_SHA256 [--cache DIR]
jetlink-models import    PATH [--name NAME] [--cache DIR]
jetlink-models inventory [--json] [--cache DIR]
jetlink-models rm        SHA256 [--artifacts] [--model] [--cache DIR]
jetlink-models prepare   REF_OR_SHA256 [--backend auto] [--device auto] [--cache DIR]
```

- Run them in the Python environment Jetlink is installed in, or use
  `python -m jetlink.registry` instead of `jetlink-models`.
- Square brackets mark optional arguments. Replace uppercase placeholders such
  as `REF` and `PATH`, without the brackets.
- The output below is shortened.

### list

Lists models, newest first, with their download or preparation status. Uses the
cached catalog if under an hour old; `--refresh` fetches a new one.

```bash
jetlink-models list
```

```
  #  Model                  Ref         Size    On disk
 13  Cinque Terre V3 Model   bf3e3631b3  766 MB  prepared (default)
 12  Cinque Terre Model V2   37bfa1413e  766 MB  no
 11  BMRLNAP Model v4        f877d7a0cc  766 MB  prepared
```

The table shortens refs. `list --json` gives the full 40-character `ref` that
`fetch` and `prepare` need. It prints the `catalog` payload from the [control
protocol](control-protocol.md#the-protocol):

```json
{"fetched_at": 1757440000.0,
 "url": "https://raw.githubusercontent.com/sunnypilot/sunnypilot-models/refs/heads/gh-pages/docs/driving_models_chestnut_v26.json",
 "default_ref": "bf3e3631b3f91d92a1020a5e0dd4298b93ff4244", "error": null,
 "models": [{"name": "Cinque Terre Model V2", "short_name": "CTMV2",
             "ref": "37bfa1413edcdc2e8844984b83727c33f81d8f46", "build_time": "<ISO 8601 timestamp>",
             "index": 12, "sha256": null, "bytes": null}]}
```

`sha256` and `bytes` are null until that entry's pointer is resolved.

### resolve

Prints a ref's SHA-256 and file size. The result is cached, so later lookups
need no network.

```bash
jetlink-models resolve f877d7a0ccc3cce943c76e285214c020cd65c899
```

```
a086d5249fc308bb...  765953504 bytes
```

The real output prints the full hash.

### fetch

Downloads the ONNX for a ref or SHA-256 to a `.part` file, checks size and hash,
then renames it into place. Progress goes to standard error, one line per
percent, so the output can be piped.

```bash
jetlink-models fetch f877d7a0ccc3cce943c76e285214c020cd65c899
```

```
resolving f877d7a0ccc3cce943c76e285214c020cd65c899
downloading a086d5249fc308bb... 765953504 bytes
  1% ... 100%
verified, saved to /mnt/data/jetlink/models/a086d5249fc308bb.onnx
```

An interrupted download restarts from the beginning. Only a verified file loses
its `.part`.

### import

Adds an ONNX file you already have: hashes it, copies it into
`<cache>/models/`, and records the name for listings.

```bash
jetlink-models import ~/Downloads/big_driving_supercombo.onnx --name "My export"
```

```
hashing /home/me/Downloads/big_driving_supercombo.onnx
a086d5249fc308bb...  765953504 bytes
copied to /mnt/data/jetlink/models/a086d5249fc308bb.onnx
```

### inventory

Lists downloaded models, prepared engines with their backends and devices, and
disk use.

```bash
jetlink-models inventory
```

```
loaded       none
last loaded  a086d5249fc308bb

models
  a086d5249fc308bb  766 MB  BMRLNAP Model v4

engines
  a086d5249fc308bb.trt10.3.0.cuda-orin  4.1 GB  trt 10.3.0  current

disk  models 766 MB, engines 4.1 GB, 63 GB free
```

`--json` prints the control protocol's `inventory` payload, as the app receives it.

### rm

- `--model` deletes the downloaded ONNX file.
- `--artifacts` deletes all of the model's prepared engines.
- A prepared engine keeps working after its download is removed.

Replace `SHA256` with the full hash from `resolve` or `inventory --json`:

```bash
jetlink-models rm SHA256 --model
```

### prepare

Fetches the model if needed, then builds an engine for the chosen backend,
exactly as `jetlink-server --build` does. If no model is recorded as last
loaded, the server loads this one at its next startup.

**Do not run `prepare` while a `jetlink-server` is using the same cache.**
Concurrent builds are unsupported and can exhaust memory, and the command cannot
always detect a running server. On a Jetson, stop the service first:

```bash
jetlink stop
```

Run `jetlink start` when preparation finishes. Or leave the server running and
ask it to prepare over the [control channel](control-protocol.md), the only safe
way to build while it serves.

```bash
jetlink-models prepare f877d7a0ccc3cce943c76e285214c020cd65c899
```

```
building a086d5249fc308bb... with trt on cuda
  1% ... 100%
built in 166.4 s, saved to /mnt/data/jetlink/engines/a086d5249fc308bb.trt10.3.0.cuda-orin.plan
```

### Exit codes

| Code | Meaning |
| --- | --- |
| 0 | Success |
| 1 | Wrong usage, or the ref or model was not found |
| 2 | A network request failed |
| 3 | The download failed verification, by size or by hash |

## On a Jetson or an installed PC

The installer's `jetlink models` runs this CLI in the server image, on the
server's models folder. Every subcommand works. Replace `<ref>` with a
40-character ref from `list --json`:

```bash
jetlink models list --json
jetlink models fetch <ref>
```

Before `prepare`, run `jetlink stop`, then `jetlink start` afterwards, or use
the control channel.
