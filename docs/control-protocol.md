# Server control protocol

Manage a running Python server from a script. The Mac, iPhone and Android apps
run the Swift server in process: the same events and commands, without a socket.

## Starting a server with a control socket

`jetlink-server` takes two extra flags:

```
--control-socket ADDR   open a local control channel; ADDR is a filesystem path
                        (AF_UNIX) or tcp://127.0.0.1:PORT (loopback only)
--parent-pid PID        exit cleanly when this process is no longer our parent
                        (checked once a second; a dead parent means getppid() changed)
```

SIGTERM is handled like SIGINT: clean shutdown, engine released, exit code 0.

## The protocol

- Stream socket, UTF-8 JSON, one object per line, newline terminated, no pretty
  printing.
- Absent optional fields are `null`, never missing, so a decoder can be strict.
- Client to server: `{"id": <int>, "cmd": "<name>", ...arguments}`. The client
  chooses `id`, positive and increasing.
- Server to client: `{"event": "<name>", "t": <unix time, float seconds>, ...}`.
- Exactly one `reply` event per command, carrying its `id`. Any other event can
  arrive at any time, including between a command and its reply.
- Several clients may connect; every event goes to all. A client more than 1000
  queued lines behind is disconnected.

## On connect

The server sends `hello`, `server`, `link`, `engine`, `inventory`, `catalog`,
then one `download` per download in progress, so a client can show state
without sending a command.

## Example

Start a server with a socket:

```bash
jetlink-server --transport usb --control-socket /tmp/jetlink-control.sock
```

Watch events from another terminal:

```bash
nc -U /tmp/jetlink-control.sock
```

Send a command by typing a line into that connection, or pipe one in:

```bash
printf '{"id":1,"cmd":"download","ref":"f877d7a0ccc3cce943c76e285214c020cd65c899"}\n' \
  | nc -U /tmp/jetlink-control.sock
```

Download, prepare, and load a model without stopping the server (`<sha256>` is
the full SHA-256):

```bash
printf '{"id":2,"cmd":"prepare","sha256":"<sha256>","frame_skip":4}\n' \
  | socat - UNIX-CONNECT:/tmp/jetlink-control.sock
```

## Events

Angle brackets are placeholders. Timestamps are ISO 8601 strings or Unix
seconds, as each field shows.

```jsonc
{"event":"hello","t":0,"protocol":1,"pid":4242,"version":"0.2.0","python":"3.14.7",
 "platform":"darwin","cache":"/Users/me/Library/Application Support/Jetlink/cache",
 "transport":"usb","port":null}
// transport is "usb"|"tcp"; port is set for tcp.

{"event":"server","t":0,"state":"serving","detail":"","backend":"ort",
 "runtime_version":"1.29.0","device":"coreml-Apple_M1_Pro"}
// state: "serving" | "stopping". backend/runtime_version/device are what the
// hello to the comma carries.

{"event":"link","t":0,"state":"waiting","detail":"waiting for a jetlink gadget at 1209:0001","peer":null}
// state: "waiting" | "connected" | "disconnected". peer: "usb" or "host:port" when connected.
// medium, when connected and known: "usb3" | "usb2" | "usb1" | "usb" (speed unknown) | "tcp".
// It starts from what the server sees and follows the comma's hello, which names its
// link and the USB speed its controller negotiated (a phone's cable is TCP over USB),
// with one more "connected" event if that changes it. Older servers leave it out.
// Emitted on transitions only, never on every 2 s poll.

{"event":"engine","t":0,"state":"none","sha256":null,"detail":"","stage":null,"frac":0.0,"msg":"","load_only":false}
// state: "none" | "building" | "loading" | "ready" | "failed"
// stage: "upload"|"patch"|"parse"|"build"|"save"|"load"|"failed"|null, frac 0..1, msg free text.
// Emitted on every state change and on progress at most 4 times a second.

{"event":"stats","t":0,"frames":1234,"fps":19.9,"total_ms":{"mean":31.2,"p99":38.0,"max":41.5},
 "gpu_ms":{"mean":29.4},"stages_ms":{"queue":0.6,"gpu":29.4,"other":1.2,"send":0.4},
 "served_ms":{"mean":31.6,"p99":38.4,"max":41.9},"slow":0,"window_s":1.0}
// Once a second while a link is connected and at least one frame was served in
// the window. slow counts frames over 60 ms in the window. total_ms runs from a
// frame's arrival to its reply being ready; served_ms adds the reply's send.
// stages_ms are means that add up to served_ms.mean: staging the inputs, the
// model run as the backend times it, the rest of the run (a worker handoff,
// the output check), and the send.

{"event":"inventory","t":0,"loaded":"<sha256>|null","last_loaded":"<sha256>|null",
 "models":[{"sha256":"…","bytes":765953504,"path":"…/models/a086d5249fc308bb.onnx","name":"BMRLNAP Model v4","ref":"f877d7a0…|null"}],
 "artifacts":[{"sha256":"…","key":"a086d5249fc308bb.ort1.29.0.coreml-Apple_M1_Pro","path":"…/engines/a086….ortcache",
               "bytes":2300000000,"backend":"ort","runtime_version":"1.29.0","device":"coreml-Apple_M1_Pro",
               "built_at":"<ISO 8601 timestamp>","build_seconds":8.2,"checkpoint":"b9facbcc-…","current":true}],
 "disk":{"models_bytes":765953504,"engines_bytes":2300000000,"free_bytes":120000000000}}
// models: every complete <sha16>.onnx in models/ (a .part is not listed).
// artifacts: every engines/*.json sidecar with a spec.sha256, any backend.
//   current is true when the key equals this server's cache key for that sha.
// name: from the catalog or from an imported model, else null.
// Emitted on connect, after every build or load completes, after forget, after
// a download or import completes, and on the inventory command.

{"event":"catalog","t":0,"fetched_at":1757440000.0,"url":"https://…/driving_models_chestnut_v26.json",
 "default_ref":"bf3e3631b3f91d92a1020a5e0dd4298b93ff4244","error":null,
 "models":[{"name":"Cinque Terre Model V2","short_name":"CTMV2",
            "ref":"37bfa1413edcdc2e8844984b83727c33f81d8f46","build_time":"<ISO 8601 timestamp>","index":12,
            "sha256":"…|null","bytes":765950064}]}
// Newest first (index descending). sha256/bytes are null until the pointer for
// that ref has been resolved. error is set when a refresh failed and the list
// is the previous cached one (possibly empty).

{"event":"download","t":0,"sha256":"…","ref":"…|null","state":"progress","frac":0.42,
 "bytes":321000000,"total":765953504,"rate_bps":41000000.0,"detail":"","source":"https://gitlab.com/…/info/lfs"}
// state: "started" | "progress" (at most 4 a second) | "done" | "failed" | "cancelled".

{"event":"import","t":0,"path":"/Users/me/Downloads/big.onnx","state":"hashing","frac":0.3,"sha256":null,"detail":""}
// state: "hashing" | "copying" | "done" | "failed". sha256 set from "copying" on.

{"event":"reply","t":0,"id":7,"ok":true}
{"event":"reply","t":0,"id":8,"ok":false,"error":"model a086d5249fc308bb is not downloaded"}
```

## Commands

| cmd | arguments | reply extras | behaviour |
| --- | --- | --- | --- |
| `status` | | | re-sends `server`, `link`, `engine`, `inventory`, `catalog` |
| `catalog` | `refresh: bool` (default false) | `queued: true` | fetches the catalog if `refresh`, or if the cache is missing or older than 3600 s; resolves pointers for refs without one (8 in parallel); emits `catalog` when done, with `error` on failure. Never blocks the reply. |
| `download` | `ref` or `sha256` (one) | `sha256` | resolves the pointer if needed; queues a download (one at a time, FIFO); `download` events follow. Error if already downloaded or queued, or the ref is unknown. |
| `cancel_download` | `sha256` | | cancels a running or queued download, removes the `.part`, emits `download` with `cancelled` |
| `import` | `path` | `queued: true` | hashes the file (streaming), copies it to `models/<sha16>.onnx` via a `.part`, records name and size; `import` events, then `inventory` |
| `prepare` | `sha256`, `frame_skip` (default 4) | `state`; `sha256` when downloading | builds if needed, loads the engine and keeps it in memory; `state` is the engine state after. With neither artifact nor model file on disk: downloads first (joining a running download for it), replies `state: "downloading"`, builds when `download` reports `done`, unless a comma connected meanwhile and uses another model. Error if the model is not in the catalog. |
| `unload` | | | releases the loaded engine; `engine` with `none` |
| `forget` | `sha256`, `artifacts: bool`, `model: bool` | | unloads that model if loaded; `artifacts` deletes every `engines/<sha16>.*`, `model` deletes `models/<sha16>.onnx` (and `.part`); removes `last-loaded.json` if it names this sha and its artifact is gone; then `inventory` |
| `inventory` | | | emits `inventory` |
| `shutdown` | | | replies, emits `server` with `stopping`, exits cleanly as on SIGINT |

- Errors are plain English sentences in `error`.
- Unknown `cmd`: `ok:false`.
- Malformed line (not JSON, or no `id`):
  `{"event":"reply","id":null,"ok":false,"error":"…"}`.
