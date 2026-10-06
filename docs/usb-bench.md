# USB and comma resource benchmarks

Run these with the comma parked and offroad. The recording bench uses private
Params and messaging, leaves the gadget owner running, and never starts controls
or pandad. A real ignition change stops it. Camera processes must not already be
running. Use a checkout of the candidate package; the bench imports that package
even when the installed openpilot checkout contains another Jetlink revision.

## Recording workload

From the comma's openpilot checkout:

```sh
OUTPUT=/data/tmp/jetlink-recording-test \
  jetlink/scripts/comma/jetlink_live_bench.sh 180 --record --resources
```

Set `OPENPILOT=/data/openpilot` when staging the Jetlink checkout elsewhere.
The output directory must not exist. The bench stops before free disk space
falls below 2 GiB; recordings remain in the output directory for inspection.

`--record` runs camerad, modeld, driver monitoring, encoderd, loggerd and
logmessaged. Python logs and video files stay under the output directory.
A run of at least 150 seconds must produce two complete recording segments,
each with an rlog and all three camera streams. Inspect `summary.json`,
`frames.csv` and the per-process logs; the summary includes startup and separate
small/large-model statistics, not just the last few frames.

`--resources` samples the child processes once a second into `resources.jsonl`:
CPU ticks, faults, per-thread scheduling and context switches, VM counters,
memory totals and free page orders. PSS is sampled every ten seconds, using
`smaps` when the kernel lacks `smaps_rollup`. `sampler_ms` records collection
cost. Convert CPU tick deltas using the recorded clock frequency and elapsed
monotonic time; 100% means one whole core. Compare matched steady-state windows.

`--write-chunk 16384` or `--write-chunk 32768` overrides the candidate's write
quantum. An unchanged older checkout can override this internally, so check its
code and actual send diagnostics before calling it a small-write baseline.
`--small` measures the small model. Keep model, host, recording and power settings
matched when comparing results.

## Byte integrity under signals

The separate diagnostic replaces inference with a SHA-256 reply. It checks every
payload byte, sequence and padding boundary. Frame and chunk positions are
stamped into random data, so replaying a previous chunk changes the digest.
The comma's main thread also receives frequent SIGUSR2 signals to exercise the
signal masking that prevents interrupted FunctionFS writes from replaying bytes.

The host needs Python's `libusb1` package and native libusb. Stop the normal
server so it cannot claim the interface, and arrange to restart it even if the
test fails. On a Linux host with the installed service, for example:

```sh
bash -c 'trap "sudo systemctl start jetlink-server.service" EXIT
  sudo systemctl stop jetlink-server.service
  sudo timeout 1800 python3 scripts/comma/jetlink_usb_integrity.py --host --frames 100000'
```

Within 45 seconds, run the same checkout's diagnostic on the comma:

```sh
PYTHONPATH=/data/openpilot /usr/local/venv/bin/python3 \
  jetlink/scripts/comma/jetlink_usb_integrity.py --frames 100000 --write-chunk 32768
```

The comma borrows the gadget from its owner and releases it on completion or
exception. No modeld may run alongside it. Match `--frames` and `--payload-bytes`
on both ends; the default payload is 393,216 bytes. This test is intentionally
separate from timing runs: hashing, signal generation and synchronous Python
host reads change the workload. It does not establish inference-output parity.

## What the fix changes and what still needs qualification

FunctionFS in the AGNOS 4.9 kernel allocates a contiguous buffer for the sum of
each writev's iovecs. Sending a complete inference at once therefore needs an
order-7 allocation. Smaller writes bound that allocation before reclaim starts;
shrinking only after ENOMEM cannot prevent the preceding stall. A live kernel
trace is needed to confirm the effective allocation, including any platform
padding. On the tested AGNOS 19.7 mici build, 32 KiB requests allocated exactly
32 KiB (order 3), with 16 KiB for the final aligned tail.

The transport masks signals and arms its write watchdog once per complete
message, retains one absolute deadline, and keeps an ENOMEM shrink for the
session. The 200 ms inference deadline and warm small-model fallback remain.
The engagement polling watcher was removed earlier; this USB write watchdog
still provides recovery from blocked endpoint I/O.

`jetlinkSend` diagnostics are logged once a second and the failed send is logged
on error. `last_send` describes one message; `totals` retains counts and maxima
for the whole transport session. A maximum write time includes kernel allocation
and transfer, not just time waiting for a host reader.

Smaller receive reserves save 256 KiB per transport and lower the queue ceiling
from 8 MiB to 512 KiB. This is not a claim of a 7.5 MiB steady-state PSS saving.
USB warp readback reuses its cached host destination while retaining tinygrad's
copy and synchronization; the phone cable path keeps its GPU mapping.

Before promoting a fork pin, qualify the reporter's device and host, USB 2/3,
uploads and model geometries; run recording/memory-pressure soaks and repeated
reconnects; test host stalls and cable loss; compare complete-run latency, CPU
and PSS. A clean short recording run or a digest-only test cannot replace those
checks. Keep measured regressions visible: the first 32 KiB recording trial was
1.44 ms slower at p50 than the 512 KiB baseline, outside the proposed +1 ms gate.

On the same bench, the candidate with cached readback completed 3,432 large-model
frames with zero drops or invalid frames, p50 24.84 ms, p99.9 28.57 ms and maximum
34.84 ms. modeld used 21.26% of one core and median PSS was 238.17 MiB. The
subsequent unchanged-code recording run reproduced a 200 ms watchdog disconnect
and 30 order-7 FunctionFS allocation warnings; its initial fallback frame took
315 ms. These observations support the allocation diagnosis, but do not close
the remaining latency or cross-device qualification gates.
