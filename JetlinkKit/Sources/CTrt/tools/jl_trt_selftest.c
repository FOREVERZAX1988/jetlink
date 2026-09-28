// Checks the shim end to end on a real GPU, the way the server uses it:
// builds a tiny ONNX, loads the plan from an mmap, runs it with a captured
// CUDA graph and a looped state (the in-graph device copy, the reply event),
// resets the state, and times a replay. Every step prints PASS or FAIL; the
// exit status is the verdict. Built against the fake (JL_TRT_FAKE) too, where
// the fake's default plan is this model, so the program itself is tested
// without a GPU; that build also checks what only the fake can stage.
//
//   jl_trt_selftest [--device N] [--keep DIR]
#define _XOPEN_SOURCE 700
// mkdtemp, which macOS hides from a strict POSIX build
#define _DARWIN_C_SOURCE

#include "jl_trt.h"
#ifdef JL_TRT_FAKE
#include "jl_trt_fake.h"
#endif

#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

// y = Cast<float>(x + state), next_state = state + 1, all [1, 8], x and
// state float16; opset 17, IR 8. Made with the onnx package:
//
//   g = helper.make_graph(
//       [helper.make_node('Add', ['x', 'state'], ['s']),
//        helper.make_node('Cast', ['s'], ['y'], to=TensorProto.FLOAT),
//        helper.make_node('Add', ['state', 'one'], ['next_state'])], 'g',
//       [info('x', FLOAT16, [1, 8]), info('state', FLOAT16, [1, 8])],
//       [info('y', FLOAT, [1, 8]), info('next_state', FLOAT16, [1, 8])],
//       [numpy_helper.from_array(np.array([1.0], np.float16), 'one')])
//   helper.make_model(g, opset_imports=[helper.make_opsetid('', 17)], ir_version=8)
static const unsigned char model[] = {
    0x08, 0x08, 0x3a, 0xbf, 0x01, 0x0a, 0x12, 0x0a, 0x01, 0x78, 0x0a, 0x05, 0x73, 0x74, 0x61, 0x74,
    0x65, 0x12, 0x01, 0x73, 0x22, 0x03, 0x41, 0x64, 0x64, 0x0a, 0x17, 0x0a, 0x01, 0x73, 0x12, 0x01,
    0x79, 0x22, 0x04, 0x43, 0x61, 0x73, 0x74, 0x2a, 0x09, 0x0a, 0x02, 0x74, 0x6f, 0x18, 0x01, 0xa0,
    0x01, 0x02, 0x0a, 0x1d, 0x0a, 0x05, 0x73, 0x74, 0x61, 0x74, 0x65, 0x0a, 0x03, 0x6f, 0x6e, 0x65,
    0x12, 0x0a, 0x6e, 0x65, 0x78, 0x74, 0x5f, 0x73, 0x74, 0x61, 0x74, 0x65, 0x22, 0x03, 0x41, 0x64,
    0x64, 0x12, 0x01, 0x67, 0x2a, 0x0d, 0x08, 0x01, 0x10, 0x0a, 0x42, 0x03, 0x6f, 0x6e, 0x65, 0x4a,
    0x02, 0x00, 0x3c, 0x5a, 0x13, 0x0a, 0x01, 0x78, 0x12, 0x0e, 0x0a, 0x0c, 0x08, 0x0a, 0x12, 0x08,
    0x0a, 0x02, 0x08, 0x01, 0x0a, 0x02, 0x08, 0x08, 0x5a, 0x17, 0x0a, 0x05, 0x73, 0x74, 0x61, 0x74,
    0x65, 0x12, 0x0e, 0x0a, 0x0c, 0x08, 0x0a, 0x12, 0x08, 0x0a, 0x02, 0x08, 0x01, 0x0a, 0x02, 0x08,
    0x08, 0x62, 0x13, 0x0a, 0x01, 0x79, 0x12, 0x0e, 0x0a, 0x0c, 0x08, 0x01, 0x12, 0x08, 0x0a, 0x02,
    0x08, 0x01, 0x0a, 0x02, 0x08, 0x08, 0x62, 0x1c, 0x0a, 0x0a, 0x6e, 0x65, 0x78, 0x74, 0x5f, 0x73,
    0x74, 0x61, 0x74, 0x65, 0x12, 0x0e, 0x0a, 0x0c, 0x08, 0x0a, 0x12, 0x08, 0x0a, 0x02, 0x08, 0x01,
    0x0a, 0x02, 0x08, 0x08, 0x42, 0x04, 0x0a, 0x00, 0x10, 0x11,
};

static int failures;
static char err[8192];

static void report(int ok, const char *step, const char *fmt, ...) __attribute__((format(printf, 3, 4)));

static void report(int ok, const char *step, const char *fmt, ...) {
  char detail[512];
  va_list args;
  va_start(args, fmt);
  vsnprintf(detail, sizeof detail, fmt, args);
  va_end(args);
  printf("%s %s: %s\n", ok ? "PASS" : "FAIL", step, detail);
  fflush(stdout);
  if (!ok) {
    failures++;
  }
}

// Stops the run: the steps after this one need it.
#define REQUIRE(call, step)                                                   \
  do {                                                                        \
    int rc_ = (call);                                                         \
    if (rc_ != JL_TRT_OK) {                                                   \
      report(0, step, "%s returned %d: %s", #call, rc_, err);                 \
      goto done;                                                              \
    }                                                                         \
  } while (0)

static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static uint16_t f16(float v) {
  // exact for the small integers and halves used here
  if (v == 0) {
    return 0;
  }
  uint32_t bits;
  memcpy(&bits, &v, sizeof bits);
  uint32_t exp = ((bits >> 23) & 0xffu) - 127 + 15;
  return (uint16_t)(((bits >> 16) & 0x8000u) | (exp << 10) | ((bits >> 13) & 0x3ffu));
}

static void log_line(void *ctx, int severity, const char *message) {
  (void)ctx;
  static const char *names[] = {"internal error", "error", "warning", "info", "verbose"};
  printf("     tensorrt %s: %s\n", severity >= 0 && severity <= 4 ? names[severity] : "?", message);
}

typedef struct {
  int roots, steps;
  char first_root[128];
} progress_seen;

static void on_progress(void *ctx, int event, const char *phase, const char *parent, int value) {
  progress_seen *p = ctx;
  if (event == JL_TRT_PHASE_START && parent == NULL) {
    if (p->roots++ == 0) {
      snprintf(p->first_root, sizeof p->first_root, "%s", phase);
    }
    printf("     phase %s (%d steps)\n", phase, value);
  } else if (event == JL_TRT_PHASE_STEP) {
    p->steps++;
  }
}

enum { X, STATE, Y, NEXT, N_TENSORS };
static const char *names[N_TENSORS] = {"x", "state", "y", "next_state"};
static const int want_input[N_TENSORS] = {1, 1, 0, 0};
static const int want_type[N_TENSORS] = {JL_TRT_FLOAT16, JL_TRT_FLOAT16, JL_TRT_FLOAT, JL_TRT_FLOAT16};
static const size_t bytes[N_TENSORS] = {16, 16, 32, 16};

static void set_x(uint16_t *x, float base) {
  for (int j = 0; j < 8; j++) {
    x[j] = f16(base + (float)j * 0.5f);
  }
}

static int y_is(const float *y, float base, float state) {
  for (int j = 0; j < 8; j++) {
    if (y[j] != base + (float)j * 0.5f + state) {
      printf("     y[%d] = %g, want %g\n", j, (double)y[j], (double)(base + (float)j * 0.5f + state));
      return 0;
    }
  }
  return 1;
}

static int write_all(const char *path, const void *data, size_t size) {
  FILE *f = fopen(path, "wb");
  if (f == NULL) {
    return 0;
  }
  int ok = fwrite(data, 1, size, f) == size;
  return fclose(f) == 0 && ok;
}

#ifdef JL_TRT_FAKE
// What only the fake can stage: a GPU fault, injected failures, and a plan
// from another TensorRT build.
static void fake_checks(void) {
  static const char *plan = "jl_trt_fake_plan 1\nbuilt 10.3.0.30\ninput x float16 1 8\ninput state float16 1 8\n"
                            "output y float32 1 8\noutput next_state float16 1 8 from state\n";
  jl_trt *t = NULL;
  jl_trt_engine *engine = NULL;
  jl_trt_context *context = NULL;
  jl_trt_stream *stream = NULL;
  jl_trt_graph *graph = NULL;
  jl_trt_graph_exec *exec = NULL;
  jl_trt_dptr dev[N_TENSORS] = {0};
  void *host[N_TENSORS] = {0};
  const char *step = "fake";
  REQUIRE(jl_trt_fake_open(NULL, &t, err, sizeof err), step);
  REQUIRE(jl_trt_engine_deserialize(t, plan, strlen(plan), &engine, err, sizeof err), step);
  REQUIRE(jl_trt_context_create(engine, &context, err, sizeof err), step);
  REQUIRE(jl_trt_stream_create(t, &stream, err, sizeof err), step);
  for (int i = 0; i < N_TENSORS; i++) {
    REQUIRE(jl_trt_mem_alloc(t, bytes[i], &dev[i], err, sizeof err), step);
    REQUIRE(jl_trt_host_alloc(t, bytes[i], &host[i], err, sizeof err), step);
    REQUIRE(jl_trt_context_set_address(context, names[i], dev[i], err, sizeof err), step);
  }
  REQUIRE(jl_trt_context_enqueue(context, stream, err, sizeof err), step);
  REQUIRE(jl_trt_capture_begin(t, stream, err, sizeof err), step);
  REQUIRE(jl_trt_copy_h2d(t, dev[X], host[X], bytes[X], stream, err, sizeof err), step);
  REQUIRE(jl_trt_context_enqueue(context, stream, err, sizeof err), step);
  REQUIRE(jl_trt_capture_end(t, stream, &graph, err, sizeof err), step);
  REQUIRE(jl_trt_graph_instantiate(t, graph, &exec, err, sizeof err), step);

  jl_trt_fake_fail(t, "graph_launch", 1, JL_TRT_CUDA_ERROR, "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: too many resources");
  int rc = jl_trt_graph_launch(t, exec, stream, err, sizeof err);
  int again = jl_trt_graph_launch(t, exec, stream, err, sizeof err);
  report(rc == JL_TRT_CUDA_ERROR && again == JL_TRT_OK && !jl_trt_sticky(t), "fake: injected error", "the next launch works");

  // closing out of order: a buffer the graph uses, freed before the graph
  jl_trt_mem_free(t, dev[X]);
  rc = jl_trt_graph_launch(t, exec, stream, err, sizeof err);
  again = jl_trt_stream_sync(t, stream, err, sizeof err);
  report(rc == JL_TRT_CUDA_STICKY && again == JL_TRT_CUDA_STICKY && jl_trt_sticky(t), "fake: replay into freed memory",
         "sticky, and latched: %s", err);

  jl_trt_fake_config other;
  jl_trt_fake_defaults(&other);
  other.minor = 16;
  other.patch = 2;
  other.build = 10;
  jl_trt *newer = NULL;
  jl_trt_engine *refused = NULL;
  REQUIRE(jl_trt_fake_open(&other, &newer, err, sizeof err), step);
  rc = jl_trt_engine_deserialize(newer, plan, strlen(plan), &refused, err, sizeof err);
  report(rc == JL_TRT_ERROR && refused == NULL && !jl_trt_sticky(newer), "fake: another build's plan", "%s", err);
  jl_trt_close(newer);
done:
  jl_trt_close(t);
}
#endif

int main(int argc, char **argv) {
  int device = 0;
  const char *keep = NULL;
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
      device = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--keep") == 0 && i + 1 < argc) {
      keep = argv[++i];
    } else {
      fprintf(stderr, "usage: %s [--device N] [--keep DIR]\n", argv[0]);
      return 2;
    }
  }
  char dir[512];
  if (keep != NULL) {
    snprintf(dir, sizeof dir, "%s", keep);
    mkdir(dir, 0755);
  } else {
    snprintf(dir, sizeof dir, "%s/jl_trt_selftest.XXXXXX", getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp");
    if (mkdtemp(dir) == NULL) {
      perror("mkdtemp");
      return 2;
    }
  }
  char onnx_path[600], plan_path[600], cache_path[600];
  snprintf(onnx_path, sizeof onnx_path, "%s/selftest.onnx", dir);
  snprintf(plan_path, sizeof plan_path, "%s/selftest.plan", dir);
  snprintf(cache_path, sizeof cache_path, "%s/timing.cache", dir);

  jl_trt *t = NULL;
  jl_trt_build *build = NULL;
  jl_trt_engine *engine = NULL;
  jl_trt_context *context = NULL;
  jl_trt_stream *stream = NULL;
  jl_trt_event *reply = NULL, *start = NULL, *end = NULL;
  jl_trt_graph *graph = NULL;
  jl_trt_graph_exec *exec = NULL;
  jl_trt_dptr dev[N_TENSORS] = {0};
  void *host[N_TENSORS] = {0};
  jl_trt_info info;

  jl_trt_get_info(NULL, &info);
  printf("jl_trt_selftest: shim compiled against TensorRT %d.%d.%d.%d (%s)\n", info.header_major, info.header_minor,
         info.header_patch, info.header_build, info.strongly_typed ? "strongly typed" : "weakly typed + FP16");

  // 1. open
#ifdef JL_TRT_FAKE
  REQUIRE(jl_trt_fake_open(NULL, &t, err, sizeof err), "open");
#else
  REQUIRE(jl_trt_open(device, &t, err, sizeof err), "open");
#endif
  jl_trt_set_logger(t, JL_TRT_LOG_WARNING, log_line, NULL);
  jl_trt_get_info(t, &info);
  report(info.major == info.header_major, "open",
         "TensorRT %d.%d.%d.%d, CUDA driver %d.%d, device %d %s sm%d%d, plugins %s", info.major, info.minor, info.patch,
         info.build, info.cuda_driver / 1000, info.cuda_driver % 1000 / 10, info.device, info.device_name, info.cc_major,
         info.cc_minor, info.plugins ? "registered" : "absent");

  // 2. build, as TrtBackend does
  {
    if (!write_all(onnx_path, model, sizeof model)) {
      report(0, "build", "cannot write %s", onnx_path);
      goto done;
    }
    progress_seen seen = {0, 0, {0}};
    double t0 = now();
    REQUIRE(jl_trt_build_create(t, &build, err, sizeof err), "build");
    REQUIRE(jl_trt_build_parse(build, onnx_path, err, sizeof err), "build");
    int layers = jl_trt_build_layers(build);
    if (!info.strongly_typed) {
      REQUIRE(jl_trt_build_set_fp16(build, err, sizeof err), "build");
    }
    jl_trt_build_set_optimization_level(build, 3);
    jl_trt_build_set_workspace(build, (size_t)256 << 20);
    jl_trt_build_set_progress(build, on_progress, &seen);
    REQUIRE(jl_trt_build_set_timing_cache(build, NULL, 0, err, sizeof err), "build");
    REQUIRE(jl_trt_build_write_plan(build, plan_path, err, sizeof err), "build");
    REQUIRE(jl_trt_build_write_timing_cache(build, cache_path, err, sizeof err), "build");
    jl_trt_build_destroy(build);
    build = NULL;
    report(layers > 0 && seen.roots > 0, "build", "%d layers, %d root phases (first \"%s\"), %d steps, %.1f s", layers,
           seen.roots, seen.first_root, seen.steps, now() - t0);
  }

  // 3. load from an mmap, and the IO it declares
  {
    int fd = open(plan_path, O_RDONLY);
    struct stat st;
    if (fd < 0 || fstat(fd, &st) != 0 || st.st_size == 0) {
      report(0, "load", "cannot open %s", plan_path);
      goto done;
    }
    void *plan = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (plan == MAP_FAILED) {
      report(0, "load", "cannot map %s", plan_path);
      goto done;
    }
    int rc = jl_trt_engine_deserialize(t, plan, (size_t)st.st_size, &engine, err, sizeof err);
    munmap(plan, (size_t)st.st_size);
    REQUIRE(rc, "load");
    int ok = jl_trt_engine_io_count(engine) == N_TENSORS;
    for (int i = 0; ok && i < N_TENSORS; i++) {
      const char *name;
      int is_input, type, rank, found = 0;
      int64_t dims[JL_TRT_MAX_DIMS];
      for (int k = 0; k < N_TENSORS; k++) {
        REQUIRE(jl_trt_engine_io(engine, k, &name, &is_input, &type, dims, &rank, err, sizeof err), "load");
        if (strcmp(name, names[i]) == 0) {
          found = 1;
          ok = ok && is_input == want_input[i] && type == want_type[i] && rank == 2 && dims[0] == 1 && dims[1] == 8;
        }
      }
      ok = ok && found;
    }
    report(ok, "load", "%lld-byte plan, x/state float16 in, y float32 and next_state float16 out", (long long)st.st_size);
    if (!ok) {
      goto done;
    }
  }

  // 4. bind, as TrtEngine does: the looped pair has no host buffer
  REQUIRE(jl_trt_context_create(engine, &context, err, sizeof err), "bind");
  REQUIRE(jl_trt_stream_create(t, &stream, err, sizeof err), "bind");
  REQUIRE(jl_trt_event_create(t, JL_TRT_EVENT_BLOCKING_SYNC | JL_TRT_EVENT_DISABLE_TIMING, &reply, err, sizeof err),
          "bind");
  for (int i = 0; i < N_TENSORS; i++) {
    REQUIRE(jl_trt_mem_alloc(t, bytes[i], &dev[i], err, sizeof err), "bind");
    if (i == X || i == Y) {
      REQUIRE(jl_trt_host_alloc(t, bytes[i], &host[i], err, sizeof err), "bind");
    }
    REQUIRE(jl_trt_context_set_address(context, names[i], dev[i], err, sizeof err), "bind");
  }
  report(1, "bind", "device buffers for 4 tensors, pinned for x and y");

  // 5. the warm run: zeroed state, one plain enqueue
#define ENQUEUE(with_reply)                                                                                  \
  do {                                                                                                       \
    REQUIRE(jl_trt_copy_h2d(t, dev[X], host[X], bytes[X], stream, err, sizeof err), step);                   \
    REQUIRE(jl_trt_context_enqueue(context, stream, err, sizeof err), step);                                 \
    REQUIRE(jl_trt_copy_d2h(t, host[Y], dev[Y], bytes[Y], stream, err, sizeof err), step);                   \
    if (with_reply) {                                                                                        \
      REQUIRE(jl_trt_event_record(t, reply, stream, JL_TRT_RECORD_EXTERNAL, err, sizeof err), step);         \
    }                                                                                                        \
    REQUIRE(jl_trt_copy_d2d(t, dev[STATE], dev[NEXT], bytes[STATE], stream, err, sizeof err), step);         \
  } while (0)
  {
    const char *step = "warm";
    REQUIRE(jl_trt_memset(t, dev[STATE], 0, bytes[STATE], stream, err, sizeof err), step);
    set_x(host[X], 1);
    ENQUEUE(0);
    REQUIRE(jl_trt_stream_sync(t, stream, err, sizeof err), step);
    report(y_is(host[Y], 1, 0), step, "y = x on a zeroed state");
  }

  // 6. capture the frame into a graph
  {
    const char *step = "capture";
    REQUIRE(jl_trt_capture_begin(t, stream, err, sizeof err), step);
    ENQUEUE(1);
    REQUIRE(jl_trt_capture_end(t, stream, &graph, err, sizeof err), step);
    REQUIRE(jl_trt_graph_instantiate(t, graph, &exec, err, sizeof err), step);
    jl_trt_graph_destroy(t, graph);
    graph = NULL;
    report(1, step, "H2D, enqueueV3, D2H, reply event, state D2D");
  }

  // 7. replays: the state advances on the GPU and shows in y
  {
    const char *step = "replay";
    int ok = 1;
    double t0 = now();
    for (int frame = 1; frame <= 5; frame++) {
      set_x(host[X], (float)frame * 10);
      REQUIRE(jl_trt_graph_launch(t, exec, stream, err, sizeof err), step);
      REQUIRE(jl_trt_event_sync(t, reply, err, sizeof err), step);
      ok = y_is(host[Y], (float)frame * 10, (float)frame) && ok;
    }
    double per = (now() - t0) / 5 * 1e3;
    REQUIRE(jl_trt_stream_sync(t, stream, err, sizeof err), step);
    report(ok, step, "5 frames, y = x + frame, %.3f ms a frame", per);
  }

  // 8. reset: a memset outside the graph zeroes the state
  {
    const char *step = "reset";
    REQUIRE(jl_trt_memset(t, dev[STATE], 0, bytes[STATE], stream, err, sizeof err), step);
    set_x(host[X], 2);
    REQUIRE(jl_trt_graph_launch(t, exec, stream, err, sizeof err), step);
    REQUIRE(jl_trt_event_sync(t, reply, err, sizeof err), step);
    report(y_is(host[Y], 2, 0), step, "y = x again");
  }

  // 9. pure GPU time from a timing pair, and a query once done
  {
    const char *step = "timing";
    float ms = -1;
    REQUIRE(jl_trt_event_create(t, JL_TRT_EVENT_BLOCKING_SYNC, &start, err, sizeof err), step);
    REQUIRE(jl_trt_event_create(t, JL_TRT_EVENT_BLOCKING_SYNC, &end, err, sizeof err), step);
    REQUIRE(jl_trt_event_record(t, start, stream, 0, err, sizeof err), step);
    REQUIRE(jl_trt_graph_launch(t, exec, stream, err, sizeof err), step);
    REQUIRE(jl_trt_event_record(t, end, stream, 0, err, sizeof err), step);
    REQUIRE(jl_trt_event_sync(t, end, err, sizeof err), step);
    REQUIRE(jl_trt_event_elapsed(t, start, end, &ms, err, sizeof err), step);
    int query = jl_trt_event_query(t, end, err, sizeof err);
    report(ms >= 0 && query == JL_TRT_OK, step, "graph %.3f ms on the GPU, query after sync %d", (double)ms, query);
  }

  // 10. memory, and a plan that is not one
  {
    size_t free_bytes = 0, total = 0;
    REQUIRE(jl_trt_mem_info(t, &free_bytes, &total, err, sizeof err), "memory");
    report(total > 0 && free_bytes <= total, "memory", "%zu MB free of %zu MB", free_bytes >> 20, total >> 20);
    jl_trt_engine *bad = NULL;
    int rc = jl_trt_engine_deserialize(t, "not a plan", 10, &bad, err, sizeof err);
    report(rc == JL_TRT_ERROR && bad == NULL && !jl_trt_sticky(t), "bad plan", "refused with %d: %s", rc, err);
  }

done:
  // TrtEngine.close's order: the stream first, since the state copy may still run
  if (stream != NULL) {
    jl_trt_stream_sync(t, stream, err, sizeof err);
  }
  jl_trt_event_destroy(t, start);
  jl_trt_event_destroy(t, end);
  jl_trt_event_destroy(t, reply);
  for (int i = 0; i < N_TENSORS; i++) {
    jl_trt_mem_free(t, dev[i]);
    jl_trt_host_free(t, host[i]);
  }
  jl_trt_graph_exec_destroy(t, exec);
  jl_trt_graph_destroy(t, graph);
  jl_trt_stream_destroy(t, stream);
  jl_trt_context_destroy(context);
  jl_trt_engine_destroy(engine);
  jl_trt_build_destroy(build);
  if (t != NULL) {
    report(!jl_trt_sticky(t), "close", "no sticky CUDA error");
#ifdef JL_TRT_FAKE
    jl_trt_fake_stats stats;
    jl_trt_fake_get_stats(t, &stats);
    report(stats.device_allocs == 0 && stats.host_allocs == 0 && stats.streams == 0 && stats.events == 0 &&
               stats.graph_execs == 0 && stats.engines == 0 && stats.contexts == 0,
           "fake", "nothing left allocated; %llu enqueues, %llu graph launches", (unsigned long long)stats.enqueues,
           (unsigned long long)stats.graph_launches);
#endif
    jl_trt_close(t);
  }
#ifdef JL_TRT_FAKE
  fake_checks();
#endif
  if (keep == NULL) {
    unlink(onnx_path);
    unlink(plan_path);
    unlink(cache_path);
    rmdir(dir);
  }
  printf("%s jl_trt_selftest: %d failure%s\n", failures ? "FAIL" : "PASS", failures, failures == 1 ? "" : "s");
  (void)device;
  return failures ? 1 : 0;
}
