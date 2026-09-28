// Tests for the fake shim itself, so the Swift tests built on it can trust
// it: the engine's arithmetic, the capture rules, the injected failures and
// the sticky latch, the builder's plan, progress and timing cache. Built and
// run by scripts/build-linux.sh fake-test until the package has a target for
// it.
#define _XOPEN_SOURCE 700

#include "jl_trt.h"
#include "jl_trt_fake.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int failures;
static char err[1024];

#define CHECK(cond)                                                    \
  do {                                                                 \
    if (!(cond)) {                                                     \
      printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);          \
      failures++;                                                      \
    }                                                                  \
  } while (0)
#define OK(call)                                                                        \
  do {                                                                                  \
    int rc_ = (call);                                                                   \
    if (rc_ != JL_TRT_OK) {                                                             \
      printf("  FAIL %s:%d: %s returned %d: %s\n", __FILE__, __LINE__, #call, rc_, err); \
      failures++;                                                                       \
    }                                                                                   \
  } while (0)

static const char *model =
    "jl_trt_fake_plan 1\n"
    "input x float16 1 8\n"
    "input state float16 1 8\n"
    "output y float32 1 8\n"
    "output next_state float16 1 8 from state\n";

static uint16_t f16(float v) {
  // exact for the small integers and halves these tests use
  if (v == 0) {
    return 0;
  }
  uint32_t bits;
  memcpy(&bits, &v, sizeof bits);
  uint32_t exp = ((bits >> 23) & 0xff) - 127 + 15;
  return (uint16_t)(((bits >> 16) & 0x8000u) | (exp << 10) | ((bits >> 13) & 0x3ffu));
}

static float f32(uint16_t h) {
  if ((h & 0x7fffu) == 0) {
    return 0;
  }
  uint32_t bits = ((uint32_t)(h & 0x8000u) << 16) | ((((h >> 10) & 0x1fu) + 112) << 23) | ((uint32_t)(h & 0x3ffu) << 13);
  float f;
  memcpy(&f, &bits, sizeof f);
  return f;
}

static jl_trt *open_fake(const jl_trt_fake_config *config) {
  jl_trt *t = NULL;
  OK(jl_trt_fake_open(config, &t, err, sizeof err));
  return t;
}

// An engine bound the way TrtEngine binds one: device and pinned buffers per
// tensor, the state pair looped on the GPU.
typedef struct {
  jl_trt *t;
  jl_trt_engine *engine;
  jl_trt_context *context;
  jl_trt_stream *stream;
  jl_trt_event *reply;
  jl_trt_dptr dev[4];
  void *host[4];
  size_t bytes[4];
  jl_trt_graph_exec *exec;
} bound;

enum { X, STATE, Y, NEXT };

static void bind_model(bound *b, jl_trt *t) {
  memset(b, 0, sizeof *b);
  b->t = t;
  OK(jl_trt_engine_deserialize(t, model, strlen(model), &b->engine, err, sizeof err));
  OK(jl_trt_context_create(b->engine, &b->context, err, sizeof err));
  OK(jl_trt_stream_create(t, &b->stream, err, sizeof err));
  OK(jl_trt_event_create(t, JL_TRT_EVENT_BLOCKING_SYNC | JL_TRT_EVENT_DISABLE_TIMING, &b->reply, err, sizeof err));
  size_t bytes[4] = {16, 16, 32, 16};
  for (int i = 0; i < 4; i++) {
    const char *name;
    int is_input, type, rank;
    int64_t dims[JL_TRT_MAX_DIMS];
    OK(jl_trt_engine_io(b->engine, i, &name, &is_input, &type, dims, &rank, err, sizeof err));
    b->bytes[i] = bytes[i];
    OK(jl_trt_mem_alloc(t, bytes[i], &b->dev[i], err, sizeof err));
    // the looped pair keeps no host buffer, as loopState frees it
    if (i == X || i == Y) {
      OK(jl_trt_host_alloc(t, bytes[i], &b->host[i], err, sizeof err));
    }
    OK(jl_trt_context_set_address(b->context, name, b->dev[i], err, sizeof err));
  }
}

// What TrtEngine's enqueue queues: the looped pair stays on the GPU.
static int enqueue(bound *b, int reply) {
  int rc;
  if ((rc = jl_trt_copy_h2d(b->t, b->dev[X], b->host[X], b->bytes[X], b->stream, err, sizeof err)) != JL_TRT_OK ||
      (rc = jl_trt_context_enqueue(b->context, b->stream, err, sizeof err)) != JL_TRT_OK ||
      (rc = jl_trt_copy_d2h(b->t, b->host[Y], b->dev[Y], b->bytes[Y], b->stream, err, sizeof err)) != JL_TRT_OK) {
    return rc;
  }
  if (reply &&
      (rc = jl_trt_event_record(b->t, b->reply, b->stream, JL_TRT_RECORD_EXTERNAL, err, sizeof err)) != JL_TRT_OK) {
    return rc;
  }
  return jl_trt_copy_d2d(b->t, b->dev[STATE], b->dev[NEXT], b->bytes[STATE], b->stream, err, sizeof err);
}

static void set_x(bound *b, float base) {
  uint16_t *x = b->host[X];
  for (int j = 0; j < 8; j++) {
    x[j] = f16(base + (float)j * 0.5f);
  }
}

static int y_is(bound *b, float base, float state) {
  const float *y = b->host[Y];
  for (int j = 0; j < 8; j++) {
    if (y[j] != base + (float)j * 0.5f + state) {
      printf("  y[%d] = %g, want %g\n", j, y[j], base + (float)j * 0.5f + state);
      return 0;
    }
  }
  return 1;
}

// In TrtEngine.close's order; the sync's error is ignored, as there.
static void unbind(bound *b) {
  jl_trt_stream_sync(b->t, b->stream, err, sizeof err);
  jl_trt_event_destroy(b->t, b->reply);
  for (int i = 0; i < 4; i++) {
    jl_trt_mem_free(b->t, b->dev[i]);
    jl_trt_host_free(b->t, b->host[i]);
  }
  jl_trt_graph_exec_destroy(b->t, b->exec);
  jl_trt_stream_destroy(b->t, b->stream);
  jl_trt_context_destroy(b->context);
  jl_trt_engine_destroy(b->engine);
}

static int no_live(jl_trt *t) {
  jl_trt_fake_stats s;
  jl_trt_fake_get_stats(t, &s);
  return s.device_allocs == 0 && s.host_allocs == 0 && s.streams == 0 && s.events == 0 && s.graphs == 0 &&
         s.graph_execs == 0 && s.engines == 0 && s.contexts == 0 && s.builds == 0;
}

static void test_open_and_info(void) {
  jl_trt *t = (jl_trt *)1;
  CHECK(jl_trt_open(0, &t, err, sizeof err) == JL_TRT_UNAVAILABLE);
  CHECK(t == NULL && strstr(err, "fake") != NULL);
  jl_trt_info info;
  jl_trt_get_info(NULL, &info);
  CHECK(info.header_major == 10 && info.major == 0 && strcmp(info.device_name, "") == 0);
  t = open_fake(NULL);
  jl_trt_get_info(t, &info);
  CHECK(info.major == 10 && info.minor == 3 && info.patch == 0 && info.build == 30);
  CHECK(strcmp(info.device_name, "Orin") == 0 && info.cc_major == 8 && info.cc_minor == 7);
  CHECK(info.strongly_typed == 0 && info.plugins == 1 && info.cuda_driver == 12060);
  size_t free_bytes = 0, total = 0;
  jl_trt_dptr p = 0;
  OK(jl_trt_mem_alloc(t, 1 << 20, &p, err, sizeof err));
  OK(jl_trt_mem_info(t, &free_bytes, &total, err, sizeof err));
  CHECK(total == (size_t)8 << 30 && free_bytes == total - (1 << 20));
  jl_trt_mem_free(t, p);
  CHECK(no_live(t));
  jl_trt_close(t);
}

static void test_io(void) {
  jl_trt *t = open_fake(NULL);
  const char *plan = "jl_trt_fake_plan 1\n# a comment\ninput img uint8 1 12 128 256\ninput dyn float16 -1 4\n"
                     "output odd other 2\n";
  jl_trt_engine *e = NULL;
  OK(jl_trt_engine_deserialize(t, plan, strlen(plan), &e, err, sizeof err));
  CHECK(jl_trt_engine_io_count(e) == 3);
  const char *name;
  int is_input, type, rank;
  int64_t dims[JL_TRT_MAX_DIMS];
  OK(jl_trt_engine_io(e, 0, &name, &is_input, &type, dims, &rank, err, sizeof err));
  CHECK(strcmp(name, "img") == 0 && is_input && type == JL_TRT_UINT8 && rank == 4 && dims[3] == 256);
  OK(jl_trt_engine_io(e, 1, &name, &is_input, &type, dims, &rank, err, sizeof err));
  CHECK(type == JL_TRT_FLOAT16 && dims[0] == -1);
  OK(jl_trt_engine_io(e, 2, &name, &is_input, &type, dims, &rank, err, sizeof err));
  CHECK(!is_input && type == 0);
  CHECK(jl_trt_engine_io(e, 3, &name, &is_input, &type, dims, &rank, err, sizeof err) == JL_TRT_ERROR);
  jl_trt_engine_destroy(e);
  CHECK(jl_trt_engine_deserialize(t, "garbage", 7, &e, err, sizeof err) == JL_TRT_ERROR && e == NULL);
  CHECK(no_live(t));
  jl_trt_close(t);
}

static void test_engine_loop_and_graph(void) {
  jl_trt *t = open_fake(NULL);
  bound b;
  bind_model(&b, t);
  jl_trt_graph *graph = NULL;

  // warm: zero the state, one plain run
  OK(jl_trt_memset(t, b.dev[STATE], 0, b.bytes[STATE], b.stream, err, sizeof err));
  set_x(&b, 1);
  OK(enqueue(&b, 0));
  OK(jl_trt_stream_sync(t, b.stream, err, sizeof err));
  CHECK(y_is(&b, 1, 0));

  // capture, then replays: the state advances on the GPU and shows in y
  OK(jl_trt_capture_begin(t, b.stream, err, sizeof err));
  OK(enqueue(&b, 1));
  OK(jl_trt_capture_end(t, b.stream, &graph, err, sizeof err));
  OK(jl_trt_graph_instantiate(t, graph, &b.exec, err, sizeof err));
  jl_trt_graph_destroy(t, graph);
  jl_trt_fake_stats s;
  jl_trt_fake_get_stats(t, &s);
  CHECK(s.captured_ops == 5 && s.enqueues == 1);
  for (int frame = 1; frame <= 3; frame++) {
    set_x(&b, (float)frame * 10);
    OK(jl_trt_graph_launch(t, b.exec, b.stream, err, sizeof err));
    OK(jl_trt_event_sync(t, b.reply, err, sizeof err));
    CHECK(y_is(&b, (float)frame * 10, (float)frame));
  }

  // reset: memset outside the graph, then the graph again
  OK(jl_trt_memset(t, b.dev[STATE], 0, b.bytes[STATE], b.stream, err, sizeof err));
  set_x(&b, 2);
  OK(jl_trt_graph_launch(t, b.exec, b.stream, err, sizeof err));
  OK(jl_trt_event_sync(t, b.reply, err, sizeof err));
  CHECK(y_is(&b, 2, 0));
  OK(jl_trt_event_query(t, b.reply, err, sizeof err));

  // the looped pair never crosses to the host
  jl_trt_fake_get_stats(t, &s);
  CHECK(s.enqueues == 5 && s.direct_enqueues == 1 && s.graph_launches == 4);
  CHECK(s.h2d == 5 && s.h2d_bytes == 5 * 16 && s.d2h == 5 && s.d2h_bytes == 5 * 32 && s.d2d == 5);
  CHECK(s.memsets == 2 && s.misuse == 0);

  // pure GPU time from a timing pair around a launch
  jl_trt_event *start = NULL, *end = NULL;
  OK(jl_trt_event_create(t, JL_TRT_EVENT_BLOCKING_SYNC, &start, err, sizeof err));
  OK(jl_trt_event_create(t, JL_TRT_EVENT_BLOCKING_SYNC, &end, err, sizeof err));
  OK(jl_trt_event_record(t, start, b.stream, 0, err, sizeof err));
  OK(jl_trt_graph_launch(t, b.exec, b.stream, err, sizeof err));
  OK(jl_trt_event_record(t, end, b.stream, 0, err, sizeof err));
  OK(jl_trt_event_sync(t, end, err, sizeof err));
  float ms = 0;
  OK(jl_trt_event_elapsed(t, start, end, &ms, err, sizeof err));
  CHECK(ms == 1.0f);
  CHECK(jl_trt_event_elapsed(t, start, b.reply, &ms, err, sizeof err) == JL_TRT_CUDA_ERROR);
  jl_trt_event_destroy(t, start);
  jl_trt_event_destroy(t, end);

  unbind(&b);
  CHECK(no_live(t));
  jl_trt_fake_get_stats(t, &s);
  CHECK(s.misuse == 0);
  jl_trt_close(t);
}

static void test_capture_rules(void) {
  jl_trt *t = open_fake(NULL);
  bound b;
  bind_model(&b, t);
  jl_trt_graph *graph = NULL;

  // a context that never ran cannot be captured
  OK(jl_trt_capture_begin(t, b.stream, err, sizeof err));
  CHECK(jl_trt_copy_h2d(t, b.dev[X], b.host[X], b.bytes[X], b.stream, err, sizeof err) == JL_TRT_OK);
  CHECK(jl_trt_context_enqueue(b.context, b.stream, err, sizeof err) == JL_TRT_ERROR);
  CHECK(jl_trt_capture_end(t, b.stream, &graph, err, sizeof err) == JL_TRT_CUDA_ERROR && graph == NULL);
  CHECK(strstr(err, "CUDA_ERROR_STREAM_CAPTURE_INVALIDATED") != NULL);

  // the stream works again after the failed capture
  OK(jl_trt_memset(t, b.dev[STATE], 0, b.bytes[STATE], b.stream, err, sizeof err));
  set_x(&b, 1);
  OK(enqueue(&b, 0));
  OK(jl_trt_stream_sync(t, b.stream, err, sizeof err));
  CHECK(y_is(&b, 1, 0));

  // anything that synchronizes on the capturing thread fails, and so does the capture
  OK(jl_trt_capture_begin(t, b.stream, err, sizeof err));
  CHECK(jl_trt_stream_sync(t, b.stream, err, sizeof err) == JL_TRT_CUDA_ERROR);
  CHECK(strstr(err, "cuStreamSynchronize: CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED") != NULL);
  CHECK(enqueue(&b, 1) == JL_TRT_CUDA_ERROR);
  CHECK(jl_trt_capture_end(t, b.stream, &graph, err, sizeof err) == JL_TRT_CUDA_ERROR);

  // so does an allocation
  jl_trt_dptr p = 0;
  OK(jl_trt_capture_begin(t, b.stream, err, sizeof err));
  CHECK(jl_trt_mem_alloc(t, 16, &p, err, sizeof err) == JL_TRT_CUDA_ERROR);
  CHECK(jl_trt_capture_end(t, b.stream, &graph, err, sizeof err) == JL_TRT_CUDA_ERROR);

  // ending what never began
  CHECK(jl_trt_capture_end(t, b.stream, &graph, err, sizeof err) == JL_TRT_CUDA_ERROR);
  CHECK(strstr(err, "CUDA_ERROR_ILLEGAL_STATE") != NULL);

  // a reply event recorded without the external flag cannot be waited on
  OK(jl_trt_capture_begin(t, b.stream, err, sizeof err));
  OK(jl_trt_copy_h2d(t, b.dev[X], b.host[X], b.bytes[X], b.stream, err, sizeof err));
  OK(jl_trt_context_enqueue(b.context, b.stream, err, sizeof err));
  OK(jl_trt_event_record(t, b.reply, b.stream, 0, err, sizeof err));
  OK(jl_trt_capture_end(t, b.stream, &graph, err, sizeof err));
  OK(jl_trt_graph_instantiate(t, graph, &b.exec, err, sizeof err));
  jl_trt_graph_destroy(t, graph);
  OK(jl_trt_graph_launch(t, b.exec, b.stream, err, sizeof err));
  CHECK(jl_trt_event_sync(t, b.reply, err, sizeof err) == JL_TRT_CUDA_ERROR);
  jl_trt_fake_stats s;
  jl_trt_fake_get_stats(t, &s);
  CHECK(s.misuse == 1 && strstr(jl_trt_fake_last_misuse(t), "EXTERNAL") != NULL);

  // an unset address fails the enqueue
  jl_trt_context *bare = NULL;
  OK(jl_trt_context_create(b.engine, &bare, err, sizeof err));
  CHECK(jl_trt_context_enqueue(bare, b.stream, err, sizeof err) == JL_TRT_ERROR);
  CHECK(strstr(err, "has no address") != NULL);
  jl_trt_context_destroy(bare);

  unbind(&b);
  CHECK(no_live(t));
  jl_trt_close(t);
}

static void test_replay_into_freed_memory_is_sticky(void) {
  jl_trt *t = open_fake(NULL);
  bound b;
  bind_model(&b, t);
  jl_trt_graph *graph = NULL;
  OK(jl_trt_memset(t, b.dev[STATE], 0, b.bytes[STATE], b.stream, err, sizeof err));
  set_x(&b, 1);
  OK(enqueue(&b, 0));
  OK(jl_trt_capture_begin(t, b.stream, err, sizeof err));
  OK(enqueue(&b, 1));
  OK(jl_trt_capture_end(t, b.stream, &graph, err, sizeof err));
  OK(jl_trt_graph_instantiate(t, graph, &b.exec, err, sizeof err));
  jl_trt_graph_destroy(t, graph);
  // closing in the wrong order: a buffer freed while the graph still runs
  jl_trt_mem_free(t, b.dev[Y]);
  b.dev[Y] = 0;
  CHECK(jl_trt_graph_launch(t, b.exec, b.stream, err, sizeof err) == JL_TRT_CUDA_STICKY);
  CHECK(strstr(err, "CUDA_ERROR_ILLEGAL_ADDRESS") != NULL);
  CHECK(jl_trt_sticky(t));
  CHECK(jl_trt_stream_sync(t, b.stream, err, sizeof err) == JL_TRT_CUDA_STICKY);
  jl_trt_fake_stats s;
  jl_trt_fake_get_stats(t, &s);
  CHECK(s.misuse == 1);
  jl_trt_close(t);
}

static void test_injected_failures(void) {
  jl_trt *t = open_fake(NULL);
  bound b;
  bind_model(&b, t);
  jl_trt_graph *graph = NULL;
  OK(jl_trt_memset(t, b.dev[STATE], 0, b.bytes[STATE], b.stream, err, sizeof err));
  set_x(&b, 1);
  OK(enqueue(&b, 0));
  OK(jl_trt_capture_begin(t, b.stream, err, sizeof err));
  OK(enqueue(&b, 1));
  OK(jl_trt_capture_end(t, b.stream, &graph, err, sizeof err));
  OK(jl_trt_graph_instantiate(t, graph, &b.exec, err, sizeof err));
  jl_trt_graph_destroy(t, graph);

  // a plain CUDA error on the second launch, then the context still works
  jl_trt_fake_fail(t, "graph_launch", 2, JL_TRT_CUDA_ERROR, "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: too many resources");
  OK(jl_trt_graph_launch(t, b.exec, b.stream, err, sizeof err));
  CHECK(jl_trt_graph_launch(t, b.exec, b.stream, err, sizeof err) == JL_TRT_CUDA_ERROR);
  CHECK(strcmp(err, "graph_launch: CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: too many resources") == 0);
  CHECK(!jl_trt_sticky(t));
  OK(jl_trt_graph_launch(t, b.exec, b.stream, err, sizeof err));

  // a sticky one latches: every later call fails without doing anything
  jl_trt_fake_fail(t, "event_sync", 1, JL_TRT_CUDA_STICKY, "CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access");
  CHECK(jl_trt_event_sync(t, b.reply, err, sizeof err) == JL_TRT_CUDA_STICKY);
  CHECK(jl_trt_sticky(t));
  CHECK(jl_trt_graph_launch(t, b.exec, b.stream, err, sizeof err) == JL_TRT_CUDA_STICKY);
  jl_trt_dptr p = 0;
  CHECK(jl_trt_mem_alloc(t, 16, &p, err, sizeof err) == JL_TRT_CUDA_STICKY);
  unbind(&b);
  CHECK(no_live(t));
  jl_trt_close(t);

  // a deserialize that fails, as a plan from another build does
  t = open_fake(NULL);
  jl_trt_engine *e = NULL;
  jl_trt_fake_fail(t, "engine_deserialize", 1, JL_TRT_ERROR, "Serialization assertion failed");
  CHECK(jl_trt_engine_deserialize(t, model, strlen(model), &e, err, sizeof err) == JL_TRT_ERROR && e == NULL);
  OK(jl_trt_engine_deserialize(t, model, strlen(model), &e, err, sizeof err));
  jl_trt_engine_destroy(e);
  jl_trt_close(t);
}

typedef struct {
  char lines[64][96];
  int n;
} record;

static void on_progress(void *ctx, int event, const char *phase, const char *parent, int value) {
  record *r = ctx;
  if (r->n < 64) {
    snprintf(r->lines[r->n++], sizeof r->lines[0], "%d %s %s %d", event, phase, parent ? parent : "-", value);
  }
}

static void on_log(void *ctx, int severity, const char *message) {
  record *r = ctx;
  if (r->n < 64) {
    snprintf(r->lines[r->n++], sizeof r->lines[0], "%d %s", severity, message);
  }
}

static char *slurp(const char *path) {
  FILE *f = fopen(path, "rb");
  if (f == NULL) {
    return NULL;
  }
  static char buf[4096];
  size_t n = fread(buf, 1, sizeof buf - 1, f);
  fclose(f);
  buf[n] = '\0';
  return buf;
}

static void test_build(const char *dir) {
  char onnx[512], plan[512], cache[512];
  snprintf(onnx, sizeof onnx, "%s/model.onnx", dir);
  snprintf(plan, sizeof plan, "%s/model.plan", dir);
  snprintf(cache, sizeof cache, "%s/timing.cache", dir);
  FILE *f = fopen(onnx, "wb");
  fputs("not really onnx", f);
  fclose(f);

  jl_trt *t = open_fake(NULL);
  record logs = {{{0}}, 0}, events = {{{0}}, 0};
  jl_trt_set_logger(t, JL_TRT_LOG_WARNING, on_log, &logs);
  jl_trt_build *b = NULL;
  OK(jl_trt_build_create(t, &b, err, sizeof err));
  CHECK(jl_trt_build_parse(b, "/nonexistent/model.onnx", err, sizeof err) == JL_TRT_ERROR);
  CHECK(strstr(err, "MODEL_DESERIALIZE_FAILED") != NULL);
  OK(jl_trt_build_parse(b, onnx, err, sizeof err));
  CHECK(jl_trt_build_layers(b) == 3);
  OK(jl_trt_build_set_fp16(b, err, sizeof err));
  jl_trt_build_set_optimization_level(b, 3);
  jl_trt_build_set_workspace(b, (size_t)256 << 20);
  jl_trt_build_set_progress(b, on_progress, &events);
  OK(jl_trt_build_set_timing_cache(b, NULL, 0, err, sizeof err));
  OK(jl_trt_build_write_plan(b, plan, err, sizeof err));
  OK(jl_trt_build_write_timing_cache(b, cache, err, sizeof err));
  jl_trt_build_destroy(b);

  const char *text = slurp(plan);
  CHECK(text != NULL && strstr(text, "built 10.3.0.30\n") != NULL);
  CHECK(text != NULL && strstr(text, "settings fp16=1 optimization_level=3 workspace=268435456 timing_cache=cold\n") != NULL);
  CHECK(strcmp(slurp(cache), "jl_trt_fake_timing 10.3.0.30 1\n") == 0);
  CHECK(logs.n == 1 && strcmp(logs.lines[0], "2 fake: building a plan") == 0);
  // root start, then per layer a nested phase of two steps and a root step
  CHECK(events.n == 2 + 3 * 5);
  CHECK(strcmp(events.lines[0], "0 fake build - 3") == 0);
  CHECK(strcmp(events.lines[1], "0 fake tactics fake build 2") == 0);
  CHECK(strcmp(events.lines[5], "1 fake build - 0") == 0);
  CHECK(strcmp(events.lines[events.n - 1], "2 fake build - 0") == 0);

  // the plan loads and runs
  char *plan_text = strdup(slurp(plan));
  jl_trt_engine *e = NULL;
  OK(jl_trt_engine_deserialize(t, plan_text, strlen(plan_text), &e, err, sizeof err));
  CHECK(e != NULL && jl_trt_engine_io_count(e) == 4);
  jl_trt_engine_destroy(e);

  // the next build is warm from the saved cache
  char *cache_text = strdup(slurp(cache));
  OK(jl_trt_build_create(t, &b, err, sizeof err));
  OK(jl_trt_build_parse(b, onnx, err, sizeof err));
  OK(jl_trt_build_set_timing_cache(b, cache_text, strlen(cache_text), err, sizeof err));
  OK(jl_trt_build_write_plan(b, plan, err, sizeof err));
  CHECK(strstr(slurp(plan), "fp16=0 optimization_level=3 workspace=0 timing_cache=warm") != NULL);
  jl_trt_build_destroy(b);
  CHECK(no_live(t));
  jl_trt_close(t);

  // another TensorRT build: the plan and the cache are both refused, and an
  // empty cache attaches in its place
  jl_trt_fake_config c;
  jl_trt_fake_defaults(&c);
  c.major = 10;
  c.minor = 16;
  c.patch = 2;
  c.build = 10;
  t = open_fake(&c);
  logs.n = 0;
  jl_trt_set_logger(t, JL_TRT_LOG_ERROR, on_log, &logs);
  CHECK(jl_trt_engine_deserialize(t, plan_text, strlen(plan_text), &e, err, sizeof err) == JL_TRT_ERROR);
  CHECK(strstr(err, "built by TensorRT 10.3.0.30, this is 10.16.2.10") != NULL && logs.n == 1);
  OK(jl_trt_build_create(t, &b, err, sizeof err));
  CHECK(jl_trt_build_set_timing_cache(b, cache_text, strlen(cache_text), err, sizeof err) == JL_TRT_ERROR);
  OK(jl_trt_build_set_timing_cache(b, NULL, 0, err, sizeof err));
  CHECK(jl_trt_build_write_plan(b, plan, err, sizeof err) == JL_TRT_ERROR);
  OK(jl_trt_build_parse(b, onnx, err, sizeof err));
  OK(jl_trt_build_write_plan(b, plan, err, sizeof err));
  CHECK(strstr(slurp(plan), "built 10.16.2.10\n") != NULL);
  jl_trt_build_destroy(b);
  jl_trt_close(t);
  free(plan_text);
  free(cache_text);

  // TensorRT 11: no FP16 flag
  jl_trt_fake_defaults(&c);
  c.major = c.header_major = 11;
  c.strongly_typed = 1;
  t = open_fake(&c);
  OK(jl_trt_build_create(t, &b, err, sizeof err));
  CHECK(jl_trt_build_set_fp16(b, err, sizeof err) == JL_TRT_ERROR);
  jl_trt_build_destroy(b);
  jl_trt_close(t);

  // no parser library
  jl_trt_fake_defaults(&c);
  c.no_parser = 1;
  t = open_fake(&c);
  CHECK(jl_trt_build_create(t, &b, err, sizeof err) == JL_TRT_UNAVAILABLE && b == NULL);
  jl_trt_close(t);
  unlink(onnx);
  unlink(plan);
  unlink(cache);
}

static void test_misuse(void) {
  jl_trt *t = open_fake(NULL);
  jl_trt_engine *e = NULL;
  jl_trt_context *c = NULL;
  OK(jl_trt_engine_deserialize(t, model, strlen(model), &e, err, sizeof err));
  OK(jl_trt_context_create(e, &c, err, sizeof err));
  jl_trt_engine_destroy(e);
  jl_trt_fake_stats s;
  jl_trt_fake_get_stats(t, &s);
  CHECK(s.misuse == 1 && strstr(jl_trt_fake_last_misuse(t), "before its execution context") != NULL);
  jl_trt_host_free(t, &s);
  jl_trt_fake_get_stats(t, &s);
  CHECK(s.misuse == 2);
  // a copy outside live device memory is refused, as CUDA does
  jl_trt_stream *stream = NULL;
  void *host = NULL;
  OK(jl_trt_stream_create(t, &stream, err, sizeof err));
  OK(jl_trt_host_alloc(t, 16, &host, err, sizeof err));
  CHECK(jl_trt_copy_h2d(t, (jl_trt_dptr)(uintptr_t)host, host, 16, stream, err, sizeof err) == JL_TRT_CUDA_ERROR);
  jl_trt_host_free(t, host);
  jl_trt_stream_destroy(t, stream);
  // close frees whatever is left
  jl_trt_close(t);
}

static void test_half(void) {
  jl_trt *t = open_fake(NULL);
  const char *plan = "jl_trt_fake_plan 1\ninput a float32 4\noutput b float16 4\n";
  jl_trt_engine *e = NULL;
  jl_trt_context *c = NULL;
  jl_trt_stream *s = NULL;
  jl_trt_dptr a = 0, b = 0;
  OK(jl_trt_engine_deserialize(t, plan, strlen(plan), &e, err, sizeof err));
  OK(jl_trt_context_create(e, &c, err, sizeof err));
  OK(jl_trt_stream_create(t, &s, err, sizeof err));
  OK(jl_trt_mem_alloc(t, 16, &a, err, sizeof err));
  OK(jl_trt_mem_alloc(t, 8, &b, err, sizeof err));
  OK(jl_trt_context_set_address(c, "a", a, err, sizeof err));
  OK(jl_trt_context_set_address(c, "b", b, err, sizeof err));
  // 1 + 2^-11 is a tie between 1 and the next half up: even wins; 65520 rounds to inf
  float in[4] = {1.0f + 1.0f / 2048.0f, 1.0f + 3.0f / 2048.0f, 65520.0f, 5.96046448e-08f};
  memcpy((void *)(uintptr_t)a, in, sizeof in);
  OK(jl_trt_context_enqueue(c, s, err, sizeof err));
  const uint16_t *out = (const uint16_t *)(uintptr_t)b;
  CHECK(out[0] == 0x3c00 && out[1] == 0x3c02 && out[2] == 0x7c00 && out[3] == 0x0001);
  CHECK(f32(out[1]) == 1.0f + 2.0f / 1024.0f);
  jl_trt_mem_free(t, a);
  jl_trt_mem_free(t, b);
  jl_trt_stream_destroy(t, s);
  jl_trt_context_destroy(c);
  jl_trt_engine_destroy(e);
  CHECK(no_live(t));
  jl_trt_close(t);
}

int main(int argc, char **argv) {
  const char *dir = argc > 1 ? argv[1] : "/tmp";
  struct {
    const char *name;
    void (*fn)(void);
  } tests[] = {
      {"open and info", test_open_and_info},
      {"io enumeration", test_io},
      {"engine, loop and graph", test_engine_loop_and_graph},
      {"capture rules", test_capture_rules},
      {"replay into freed memory is sticky", test_replay_into_freed_memory_is_sticky},
      {"injected failures", test_injected_failures},
      {"misuse", test_misuse},
      {"float16 rounding", test_half},
  };
  for (size_t i = 0; i < sizeof tests / sizeof tests[0]; i++) {
    int before = failures;
    tests[i].fn();
    printf("%s %s\n", failures == before ? "PASS" : "FAIL", tests[i].name);
  }
  int before = failures;
  test_build(dir);
  printf("%s build\n", failures == before ? "PASS" : "FAIL");
  printf("%s: %d failure%s\n", failures ? "FAIL" : "PASS", failures, failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
