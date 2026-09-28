// Controls for the fake shim, jl_trt_fake.c: jl_trt.h over host memory, with
// no GPU and no TensorRT, so every Swift path runs in tests on a Mac and in a
// plain Linux container. It is compiled only when TensorRT's headers are not
// supplied, and its jl_trt_open always returns JL_TRT_UNAVAILABLE: a binary
// that shipped it by mistake could never serve with it. Tests get a handle
// from jl_trt_fake_open instead.
//
// Device memory is host memory, so a jl_trt_dptr is a host address; copies
// are memcpy; streams, events and graphs run synchronously; a capture records
// what is queued and a launch replays it. New memory is filled with 0xff, so
// a float read before anything wrote it is NaN.
//
// It keeps the rules the real stack keeps, so a Swift path that breaks one
// fails here first: nothing that synchronizes or allocates on the capturing
// thread (the capture is invalidated), a context's first enqueue is not
// captured, every IO tensor has an address before an enqueue, copies stay
// inside live allocations and move pinned host memory, a graph never replays
// into freed memory (a sticky illegal address, as on a GPU). A rule broken in
// a call that returns nothing (a destroy out of order, a double free) latches
// the sticky flag, so the next call fails with a "fake:" message saying why.
//
// A plan is text:
//
//   jl_trt_fake_plan 1
//   built 10.3.0.30
//   input x float16 1 8
//   input state float16 1 8
//   output y float32 1 8
//   output next_state float16 1 8 from state
//
// `built` is optional: a plan that has it loads only on a fake of that exact
// version, as TensorRT refuses a plan from another build. Types are float32,
// float16, uint8, int8, int32, int64, bool, and `other` for a type jetlink
// does not stage (jl_trt_engine_io reports 0); a dim of -1 is dynamic. An
// enqueue computes, in double, output[j] = input[j] + 1 for an output `from`
// an input (a next_state), and otherwise output[j] = the sum over every input
// k of input_k[j mod count_k]; then converts to the output's type (float16
// rounds to nearest even). So a looped state that is never copied back still
// shows in the other outputs: after a reset, y = x, then x + 1, x + 2...
// Each enqueue adds config.enqueue_ms to the clock timing events read.
//
// A build writes the plan set by jl_trt_fake_set_build (by default the one
// above, which is also jl_trt_selftest's model), with `built` and a
// `settings` line that records what the build was given: fp16=0|1
// optimization_level=N workspace=BYTES timing_cache=none|cold|warm. It
// reports a root phase "fake build" with a step per layer, each with a nested
// phase "fake tactics" of two steps, and logs one warning. A timing cache is
// the line "jl_trt_fake_timing <major>.<minor>.<patch>.<build> <builds>";
// one of another version fails to attach, and each build adds one.
#ifndef JL_TRT_FAKE_H
#define JL_TRT_FAKE_H

#include "jl_trt.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  // What jl_trt_get_info reports.
  int major, minor, patch, build;
  int header_major, header_minor, header_patch, header_build;
  int strongly_typed;
  int cuda_driver;
  int plugins;
  int device;
  const char *device_name;
  int cc_major, cc_minor;
  // What jl_trt_mem_info reports as total; free is total less what is
  // allocated.
  size_t total_memory;
  // jl_trt_build_create returns JL_TRT_UNAVAILABLE, as without
  // libnvonnxparser.
  int no_parser;
  // What each enqueue adds to the fake GPU clock.
  float enqueue_ms;
} jl_trt_fake_config;

// TensorRT 10.3.0.30 on an "Orin", sm87, 8 GB, CUDA 12.6, weakly typed, with
// plugins and a parser; 1 ms an enqueue.
void jl_trt_fake_defaults(jl_trt_fake_config *config);

// A fake jl_trt; config NULL takes the defaults. device_name is copied.
int jl_trt_fake_open(const jl_trt_fake_config *config, jl_trt **out, char *err, size_t errlen);

// Makes the nth next call of `call` fail, 1 being the very next one. `call`
// is the function's name without "jl_trt_": "graph_launch",
// "context_enqueue", "engine_deserialize", "build_write_plan", "mem_alloc".
// It returns `code`, with err reading "<call>: <message>", and does nothing
// else; JL_TRT_CUDA_STICKY latches as the real shim's does. Calls that return
// nothing cannot fail.
void jl_trt_fake_fail(jl_trt *trt, const char *call, int nth, int code, const char *message);

// The input and output lines the next build writes into its plan, and the
// layer count jl_trt_build_layers reports once parsed.
void jl_trt_fake_set_build(jl_trt *trt, const char *tensor_lines, int layers);

typedef struct {
  // Stream-ordered work done, direct or replayed by a graph; engine runs.
  uint64_t h2d, d2h, d2d, memsets, enqueues, graph_launches;
  // Live objects: all 0 once everything is destroyed.
  int64_t device_allocs, host_allocs, streams, events, graphs, graph_execs, engines, contexts, builds;
} jl_trt_fake_stats;

void jl_trt_fake_get_stats(jl_trt *trt, jl_trt_fake_stats *out);

#ifdef __cplusplus
}
#endif

#endif
