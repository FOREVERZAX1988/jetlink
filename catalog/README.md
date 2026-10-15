# Extra big models

`extra_big_models.json` lists big models that sunnypilot's model library does
not carry yet, such as previews from comma's open pull requests. Commas without
a chestnut, and every Jetlink server, add it to sunnypilot's list. An edit on
`main` reaches them within the hour, with no release.

Each entry is a sunnypilot catalog bundle. The comma reads every field below,
and a bundle missing one makes it drop the whole list, so copy an existing entry:

| Field | Value |
| --- | --- |
| `ref` | The comma openpilot commit whose `big_driving_supercombo.onnx` is the model. Check it with `jetlink-server models resolve <ref>`. |
| `display_name` | The name in the picker, ending in the model's date as `(Month DD, YYYY)`. |
| `short_name` | The short name the comma shows for the selected model. |
| `index` | Its place in the list: higher comes first. |
| `minimum_selector_version` | `"19"`, the fork's version. |
| `is_big`, `is_20hz` | `true`. |
| `generation`, `environment`, `runner`, `build_time` | As sunnypilot's newest big model has them. |
| `overrides` | `folder` groups the picker; `lat` and `long` are the model's smoothing, as sunnypilot sets them. |
| `models` | `[]`. A chestnut has nothing to download, and does not see this list. |

Name a preview after comma's commit, with "Preview" and the commit's date:
`Mountain Dew Preview (October 08, 2026)`. Don't add version numbers comma
didn't use. When sunnypilot lists the same commit, its entry wins. When it lists
the model under another commit, remove ours. Removing an entry moves anyone who
picked it back to the default model.

## Pinned models

`pointers` pins a commit to the ONNX Jetlink runs for it, overriding what
Jetlink looked up:

```json
"pointers": {
  "<commit>": {"oid": "<the ONNX's SHA-256>", "size": <its bytes>}
}
```

Use it when a commit was looked up wrong. Every comma and Jetlink server keeps
what it looked up for good, so a fix to the lookup alone doesn't reach the ones
that already have an answer. Each one saves the pins into what it keeps when it
fetches this list, and keeps them offline after that. A comma on a fork without
sunnypilot's model manager doesn't fetch this list, so it never gets them. A pin can stay in the list
indefinitely; removing one only matters for a device that never fetched it.
Get the values from Hugging Face's listing of the export folder, or
`jetlink-server models resolve <ref>` on a fixed build.
