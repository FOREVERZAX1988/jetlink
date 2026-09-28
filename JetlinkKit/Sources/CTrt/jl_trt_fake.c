// jl_trt.h over host memory, for tests: see jl_trt_fake.h for what it models
// and the rules it keeps. Everything runs under one lock per jl_trt, a
// recursive one, so a log or progress callback may call back in.

// strtok_r, strdup and recursive mutexes are POSIX, not C11
#define _XOPEN_SOURCE 700

#include "jl_trt.h"
#include "jl_trt_fake.h"

#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_TENSORS 32
#define MAX_FAILS 16

enum { OBJ_DEVICE, OBJ_HOST, OBJ_STREAM, OBJ_EVENT, OBJ_GRAPH, OBJ_EXEC, OBJ_ENGINE, OBJ_CONTEXT, OBJ_BUILD };

enum { OP_H2D, OP_D2H, OP_D2D, OP_MEMSET, OP_ENQUEUE, OP_RECORD };

typedef struct {
  int kind;
  uintptr_t dst, src;
  size_t size;
  uint8_t value;
  jl_trt_context *context;
  jl_trt_event *event;
} op;

typedef struct {
  op *ops;
  size_t n, cap;
} op_list;

typedef struct obj {
  int kind;
  void *ptr;
  size_t size;
  struct obj *next;
} obj;

typedef struct {
  char call[48];
  int nth, code;
  char message[256];
} fail;

struct jl_trt {
  pthread_mutex_t lock;
  jl_trt_fake_config config;
  char device_name[128];
  int sticky;
  jl_trt_log_fn log;
  void *log_ctx;
  int log_min;
  fail fails[MAX_FAILS];
  int n_fails;
  jl_trt_fake_stats stats;
  obj *live;
  char misuse[512];
  // one capture at a time, and the thread that began it
  jl_trt_stream *capturing;
  pthread_t capture_thread;
  double clock_ms;
  char *build_tensors;
  int build_layers;
};

struct jl_trt_stream {
  jl_trt *trt;
  int capturing, invalidated;
  op_list ops;
};

struct jl_trt_event {
  unsigned flags;
  int recorded;
  // recorded inside a capture without JL_TRT_RECORD_EXTERNAL: not for the host
  int internal;
  double stamp;
};

struct jl_trt_graph {
  op_list ops;
};

struct jl_trt_graph_exec {
  op_list ops;
};

typedef struct {
  char name[64];
  int is_input, type, rank;
  int64_t dims[JL_TRT_MAX_DIMS];
  // the input a next_state is fed from, or -1
  int from;
} tensor;

struct jl_trt_engine {
  jl_trt *trt;
  int n;
  tensor tensors[MAX_TENSORS];
};

struct jl_trt_context {
  jl_trt_engine *engine;
  jl_trt_dptr address[MAX_TENSORS];
  int ran;
};

struct jl_trt_build {
  jl_trt *trt;
  int parsed, fp16, optimization_level;
  size_t workspace;
  jl_trt_progress_fn progress;
  void *progress_ctx;
  // -1: no timing cache attached; else the builds it holds
  int cache_builds;
};

static const char *const default_tensors =
    "input x float16 1 8\n"
    "input state float16 1 8\n"
    "output y float32 1 8\n"
    "output next_state float16 1 8 from state\n";

// --- errors, the lock, injected failures -------------------------------------

static int say(char *err, size_t errlen, int code, const char *fmt, ...) {
  if (err != NULL && errlen > 0) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(err, errlen, fmt, args);
    va_end(args);
  }
  return code;
}

static int leave(jl_trt *t, int code) {
  pthread_mutex_unlock(&t->lock);
  return code;
}

static void logf_(jl_trt *t, int severity, const char *fmt, ...) {
  if (t->log == NULL || severity > t->log_min) {
    return;
  }
  char line[512];
  va_list args;
  va_start(args, fmt);
  vsnprintf(line, sizeof line, fmt, args);
  va_end(args);
  t->log(t->log_ctx, severity, line);
}

static void misuse(jl_trt *t, const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  vsnprintf(t->misuse, sizeof t->misuse, fmt, args);
  va_end(args);
  t->stats.misuse++;
}

// Takes the lock. Anything but JL_TRT_OK has released it again, with err
// written: the latched sticky error, or a failure injected for this call.
static int enter(jl_trt *t, const char *call, char *err, size_t errlen) {
  pthread_mutex_lock(&t->lock);
  if (t->sticky) {
    return leave(t, say(err, errlen, JL_TRT_CUDA_STICKY,
                        "%s: CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered (latched)", call));
  }
  for (int i = 0; i < t->n_fails; i++) {
    fail *f = &t->fails[i];
    if (strcmp(f->call, call) != 0 || --f->nth > 0) {
      continue;
    }
    int code = f->code;
    say(err, errlen, code, "%s: %s", call, f->message);
    t->fails[i] = t->fails[--t->n_fails];
    if (code == JL_TRT_CUDA_STICKY) {
      t->sticky = 1;
    }
    return leave(t, code);
  }
  return JL_TRT_OK;
}

// A call CUDA forbids on the thread that is capturing: it fails, and so does
// the capture.
static int unsafe(jl_trt *t, const char *call, char *err, size_t errlen) {
  if (t->capturing == NULL || !pthread_equal(t->capture_thread, pthread_self())) {
    return JL_TRT_OK;
  }
  t->capturing->invalidated = 1;
  return say(err, errlen, JL_TRT_CUDA_ERROR,
             "%s: CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED: operation not permitted when stream is capturing", call);
}

static void sticky(jl_trt *t) {
  t->sticky = 1;
}

// --- live objects ----------------------------------------------------------

static void track(jl_trt *t, int kind, void *ptr, size_t size) {
  obj *o = calloc(1, sizeof(obj));
  if (o == NULL) {
    abort();
  }
  o->kind = kind;
  o->ptr = ptr;
  o->size = size;
  o->next = t->live;
  t->live = o;
}

static obj *find(jl_trt *t, int kind, const void *ptr) {
  for (obj *o = t->live; o != NULL; o = o->next) {
    if (o->kind == kind && o->ptr == ptr) {
      return o;
    }
  }
  return NULL;
}

// Whether [addr, addr + size) lies inside one live allocation of `kind`.
static int inside(jl_trt *t, int kind, uintptr_t addr, size_t size) {
  for (obj *o = t->live; o != NULL; o = o->next) {
    uintptr_t base = (uintptr_t)o->ptr;
    if (o->kind == kind && addr >= base && size <= o->size && addr - base <= o->size - size) {
      return 1;
    }
  }
  return 0;
}

static int untrack(jl_trt *t, int kind, const void *ptr) {
  for (obj **link = &t->live; *link != NULL; link = &(*link)->next) {
    if ((*link)->kind == kind && (*link)->ptr == ptr) {
      obj *o = *link;
      *link = o->next;
      free(o);
      return 1;
    }
  }
  return 0;
}

static void free_ops(op_list *list) {
  free(list->ops);
  list->ops = NULL;
  list->n = list->cap = 0;
}

static void push(op_list *list, op o) {
  if (list->n == list->cap) {
    size_t cap = list->cap ? list->cap * 2 : 16;
    op *ops = realloc(list->ops, cap * sizeof(op));
    if (ops == NULL) {
      abort();
    }
    list->ops = ops;
    list->cap = cap;
  }
  list->ops[list->n++] = o;
}

static int copy_ops(op_list *to, const op_list *from) {
  to->n = to->cap = from->n;
  to->ops = NULL;
  if (from->n > 0) {
    to->ops = malloc(from->n * sizeof(op));
    if (to->ops == NULL) {
      return 0;
    }
    memcpy(to->ops, from->ops, from->n * sizeof(op));
  }
  return 1;
}

static void destroy_obj(obj *o) {
  switch (o->kind) {
  case OBJ_STREAM:
    free_ops(&((jl_trt_stream *)o->ptr)->ops);
    break;
  case OBJ_GRAPH:
    free_ops(&((jl_trt_graph *)o->ptr)->ops);
    break;
  case OBJ_EXEC:
    free_ops(&((jl_trt_graph_exec *)o->ptr)->ops);
    break;
  default:
    break;
  }
  free(o->ptr);
  free(o);
}

// --- element types -----------------------------------------------------------

static size_t type_size(int type) {
  switch (type) {
  case JL_TRT_FLOAT:
  case JL_TRT_INT32:
    return 4;
  case JL_TRT_FLOAT16:
    return 2;
  case JL_TRT_INT64:
    return 8;
  case JL_TRT_UINT8:
  case JL_TRT_INT8:
  case JL_TRT_BOOL:
    return 1;
  default:
    return 0;
  }
}

static const struct {
  const char *name;
  int type;
} type_names[] = {
    {"float32", JL_TRT_FLOAT}, {"float16", JL_TRT_FLOAT16}, {"uint8", JL_TRT_UINT8}, {"int8", JL_TRT_INT8},
    {"int32", JL_TRT_INT32},   {"int64", JL_TRT_INT64},     {"bool", JL_TRT_BOOL},   {"other", 0},
};

static float half_to_float(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1fu;
  uint32_t mant = h & 0x3ffu;
  uint32_t bits;
  if (exp == 0 && mant == 0) {
    bits = sign;
  } else if (exp == 0) {
    exp = 113;
    while ((mant & 0x400u) == 0) {
      mant <<= 1;
      exp--;
    }
    bits = sign | (exp << 23) | ((mant & 0x3ffu) << 13);
  } else if (exp == 31) {
    bits = sign | 0x7f800000u | (mant << 13);
  } else {
    bits = sign | ((exp + 112) << 23) | (mant << 13);
  }
  float f;
  memcpy(&f, &bits, sizeof f);
  return f;
}

// Round to nearest even, as a GPU converts.
static uint16_t float_to_half(float value) {
  uint32_t x;
  memcpy(&x, &value, sizeof x);
  uint32_t sign = (x >> 16) & 0x8000u;
  uint32_t mag = x & 0x7fffffffu;
  if (mag >= 0x7f800000u) {
    return (uint16_t)(sign | 0x7c00u | (mag > 0x7f800000u ? 0x200u : 0));
  }
  if (mag >= 0x477ff000u) {
    return (uint16_t)(sign | 0x7c00u);
  }
  if (mag >= 0x38800000u) {
    uint32_t h = (mag - 0x38000000u) >> 13;
    uint32_t rest = mag & 0x1fffu;
    if (rest > 0x1000u || (rest == 0x1000u && (h & 1u))) {
      h++;
    }
    return (uint16_t)(sign | h);
  }
  if (mag < 0x33000000u) {
    return (uint16_t)sign;
  }
  uint32_t shift = 126u - (mag >> 23);
  uint32_t m = (mag & 0x7fffffu) | 0x800000u;
  uint32_t h = m >> shift;
  uint32_t rest = m & ((1u << shift) - 1u);
  uint32_t half = 1u << (shift - 1u);
  if (rest > half || (rest == half && (h & 1u))) {
    h++;
  }
  return (uint16_t)(sign | h);
}

static double load(int type, const void *base, size_t index) {
  const uint8_t *p = (const uint8_t *)base + index * type_size(type);
  switch (type) {
  case JL_TRT_FLOAT: {
    float v;
    memcpy(&v, p, sizeof v);
    return v;
  }
  case JL_TRT_FLOAT16: {
    uint16_t v;
    memcpy(&v, p, sizeof v);
    return half_to_float(v);
  }
  case JL_TRT_INT32: {
    int32_t v;
    memcpy(&v, p, sizeof v);
    return v;
  }
  case JL_TRT_INT64: {
    int64_t v;
    memcpy(&v, p, sizeof v);
    return (double)v;
  }
  case JL_TRT_UINT8:
    return *p;
  case JL_TRT_INT8:
    return (int8_t)*p;
  case JL_TRT_BOOL:
    return *p != 0;
  default:
    return 0;
  }
}

static void store(int type, void *base, size_t index, double value) {
  uint8_t *p = (uint8_t *)base + index * type_size(type);
  switch (type) {
  case JL_TRT_FLOAT: {
    float v = (float)value;
    memcpy(p, &v, sizeof v);
    break;
  }
  case JL_TRT_FLOAT16: {
    uint16_t v = float_to_half((float)value);
    memcpy(p, &v, sizeof v);
    break;
  }
  case JL_TRT_INT32: {
    int32_t v = (int32_t)value;
    memcpy(p, &v, sizeof v);
    break;
  }
  case JL_TRT_INT64: {
    int64_t v = (int64_t)value;
    memcpy(p, &v, sizeof v);
    break;
  }
  case JL_TRT_UINT8:
    *p = (uint8_t)(int64_t)value;
    break;
  case JL_TRT_INT8:
    *p = (uint8_t)(int8_t)(int64_t)value;
    break;
  case JL_TRT_BOOL:
    *p = value != 0;
    break;
  default:
    break;
  }
}

static size_t tensor_count(const tensor *x) {
  size_t n = 1;
  for (int i = 0; i < x->rank; i++) {
    n *= x->dims[i] < 0 ? 0 : (size_t)x->dims[i];
  }
  return n;
}

// --- running the engine and the work on a stream ------------------------------

// Whether every tensor has an address and the engine can run at all.
static int check_bound(jl_trt *t, jl_trt_context *c, char *err, size_t errlen) {
  jl_trt_engine *e = c->engine;
  for (int i = 0; i < e->n; i++) {
    const tensor *x = &e->tensors[i];
    if (type_size(x->type) == 0 || tensor_count(x) == 0) {
      logf_(t, JL_TRT_LOG_ERROR, "fake: tensor %s has a type or shape the fake cannot run", x->name);
      return say(err, errlen, JL_TRT_ERROR, "fake: tensor %s has a type or shape the fake cannot run", x->name);
    }
    if (c->address[i] == 0) {
      logf_(t, JL_TRT_LOG_ERROR, "IExecutionContext::enqueueV3: tensor %s has no address", x->name);
      return say(err, errlen, JL_TRT_ERROR, "IExecutionContext::enqueueV3: tensor %s has no address", x->name);
    }
  }
  return JL_TRT_OK;
}

static int illegal(jl_trt *t, const char *call, char *err, size_t errlen, const char *why) {
  misuse(t, "%s", why);
  sticky(t);
  return say(err, errlen, JL_TRT_CUDA_STICKY, "%s: CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered (fake: %s)",
             call, why);
}

static int run_engine(jl_trt *t, jl_trt_context *c, char *err, size_t errlen) {
  if (find(t, OBJ_CONTEXT, c) == NULL) {
    return illegal(t, "cuGraphLaunch", err, errlen, "a graph ran a destroyed execution context");
  }
  int rc = check_bound(t, c, err, errlen);
  if (rc != JL_TRT_OK) {
    return rc;
  }
  jl_trt_engine *e = c->engine;
  for (int i = 0; i < e->n; i++) {
    const tensor *x = &e->tensors[i];
    if (!inside(t, OBJ_DEVICE, (uintptr_t)c->address[i], tensor_count(x) * type_size(x->type))) {
      char why[160];
      snprintf(why, sizeof why, "tensor %s is not inside live device memory", x->name);
      return illegal(t, "enqueueV3", err, errlen, why);
    }
  }
  for (int o = 0; o < e->n; o++) {
    const tensor *out = &e->tensors[o];
    if (out->is_input) {
      continue;
    }
    size_t n = tensor_count(out);
    void *dst = (void *)(uintptr_t)c->address[o];
    for (size_t j = 0; j < n; j++) {
      double v = 0;
      if (out->from >= 0) {
        const tensor *in = &e->tensors[out->from];
        v = load(in->type, (void *)(uintptr_t)c->address[out->from], j % tensor_count(in)) + 1;
      } else {
        for (int i = 0; i < e->n; i++) {
          const tensor *in = &e->tensors[i];
          if (in->is_input) {
            v += load(in->type, (void *)(uintptr_t)c->address[i], j % tensor_count(in));
          }
        }
      }
      store(out->type, dst, j, v);
    }
  }
  c->ran = 1;
  t->clock_ms += t->config.enqueue_ms;
  t->stats.enqueues++;
  return JL_TRT_OK;
}

static const char *op_call(int kind) {
  switch (kind) {
  case OP_H2D:
    return "cuMemcpyHtoDAsync";
  case OP_D2H:
    return "cuMemcpyDtoHAsync";
  case OP_D2D:
    return "cuMemcpyDtoDAsync";
  case OP_MEMSET:
    return "cuMemsetD8Async";
  case OP_RECORD:
    return "cuEventRecordWithFlags";
  default:
    return "enqueueV3";
  }
}

// Does one piece of work. `replay`: from a graph, whose addresses were baked
// in, so a bad one is a GPU fault rather than a refused call.
static int execute(jl_trt *t, const op *o, int replay, char *err, size_t errlen) {
  const char *call = op_call(o->kind);
  int dst_device = o->kind == OP_H2D || o->kind == OP_D2D || o->kind == OP_MEMSET;
  int src_device = o->kind == OP_D2H || o->kind == OP_D2D;
  if (o->kind == OP_ENQUEUE) {
    return run_engine(t, o->context, err, errlen);
  }
  if (o->kind == OP_RECORD) {
    if (find(t, OBJ_EVENT, o->event) == NULL) {
      return illegal(t, call, err, errlen, "a graph recorded a destroyed event");
    }
    o->event->recorded = 1;
    o->event->internal = 0;
    o->event->stamp = t->clock_ms;
    return JL_TRT_OK;
  }
  if ((dst_device && !inside(t, OBJ_DEVICE, o->dst, o->size)) || (src_device && !inside(t, OBJ_DEVICE, o->src, o->size))) {
    if (replay) {
      return illegal(t, call, err, errlen, "a graph touched device memory that is no longer allocated");
    }
    misuse(t, "%s outside live device memory", call);
    return say(err, errlen, JL_TRT_CUDA_ERROR, "%s: CUDA_ERROR_INVALID_VALUE: invalid argument (fake: outside live device memory)",
               call);
  }
  if ((o->kind == OP_H2D && !inside(t, OBJ_HOST, o->src, o->size)) ||
      (o->kind == OP_D2H && !inside(t, OBJ_HOST, o->dst, o->size))) {
    if (replay) {
      return illegal(t, call, err, errlen, "a graph touched host memory that is no longer allocated");
    }
    misuse(t, "%s with host memory jl_trt_host_alloc did not return", call);
  }
  switch (o->kind) {
  case OP_H2D:
    memcpy((void *)o->dst, (const void *)o->src, o->size);
    t->stats.h2d++;
    t->stats.h2d_bytes += o->size;
    break;
  case OP_D2H:
    memcpy((void *)o->dst, (const void *)o->src, o->size);
    t->stats.d2h++;
    t->stats.d2h_bytes += o->size;
    break;
  case OP_D2D:
    memmove((void *)o->dst, (const void *)o->src, o->size);
    t->stats.d2d++;
    t->stats.d2d_bytes += o->size;
    break;
  case OP_MEMSET:
    memset((void *)o->dst, o->value, o->size);
    t->stats.memsets++;
    t->stats.memset_bytes += o->size;
    break;
  default:
    break;
  }
  return JL_TRT_OK;
}

// Queues work on a stream: recorded while it captures, done at once otherwise.
static int submit(jl_trt *t, jl_trt_stream *s, op o, char *err, size_t errlen) {
  if (find(t, OBJ_STREAM, s) == NULL) {
    misuse(t, "%s on a stream that is not live", op_call(o.kind));
    return say(err, errlen, JL_TRT_CUDA_ERROR, "%s: CUDA_ERROR_INVALID_HANDLE: invalid resource handle", op_call(o.kind));
  }
  if (!s->capturing) {
    return execute(t, &o, 0, err, errlen);
  }
  if (s->invalidated) {
    return say(err, errlen, JL_TRT_CUDA_ERROR,
               "%s: CUDA_ERROR_STREAM_CAPTURE_INVALIDATED: operation failed due to a previous error during capture",
               op_call(o.kind));
  }
  push(&s->ops, o);
  return JL_TRT_OK;
}

// --- library --------------------------------------------------------------------

void jl_trt_fake_defaults(jl_trt_fake_config *c) {
  memset(c, 0, sizeof *c);
  c->major = c->header_major = 10;
  c->minor = c->header_minor = 3;
  c->patch = c->header_patch = 0;
  c->build = c->header_build = 30;
  c->cuda_driver = 12060;
  c->plugins = 1;
  c->device_name = "Orin";
  c->cc_major = 8;
  c->cc_minor = 7;
  c->total_memory = (size_t)8 << 30;
  c->enqueue_ms = 1.0f;
}

int jl_trt_open(int device, jl_trt **out, char *err, size_t errlen) {
  (void)device;
  *out = NULL;
  return say(err, errlen, JL_TRT_UNAVAILABLE, "this is the fake shim, built without TensorRT");
}

int jl_trt_fake_open(const jl_trt_fake_config *config, jl_trt **out, char *err, size_t errlen) {
  *out = NULL;
  jl_trt *t = calloc(1, sizeof(jl_trt));
  if (t == NULL) {
    return say(err, errlen, JL_TRT_ERROR, "out of memory");
  }
  pthread_mutexattr_t attr;
  pthread_mutexattr_init(&attr);
  pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
  pthread_mutex_init(&t->lock, &attr);
  pthread_mutexattr_destroy(&attr);
  if (config != NULL) {
    t->config = *config;
  } else {
    jl_trt_fake_defaults(&t->config);
  }
  snprintf(t->device_name, sizeof t->device_name, "%s", t->config.device_name ? t->config.device_name : "");
  t->config.device_name = t->device_name;
  t->misuse[0] = '\0';
  t->build_layers = 3;
  *out = t;
  return JL_TRT_OK;
}

void jl_trt_close(jl_trt *t) {
  if (t == NULL) {
    return;
  }
  while (t->live != NULL) {
    obj *o = t->live;
    t->live = o->next;
    destroy_obj(o);
  }
  free(t->build_tensors);
  pthread_mutex_destroy(&t->lock);
  free(t);
}

void jl_trt_get_info(const jl_trt *t, jl_trt_info *out) {
  jl_trt_fake_config d;
  jl_trt_fake_defaults(&d);
  const jl_trt_fake_config *c = t != NULL ? &t->config : &d;
  memset(out, 0, sizeof *out);
  out->header_major = c->header_major;
  out->header_minor = c->header_minor;
  out->header_patch = c->header_patch;
  out->header_build = c->header_build;
  out->strongly_typed = c->strongly_typed;
  out->device_name = "";
  if (t == NULL) {
    return;
  }
  out->major = c->major;
  out->minor = c->minor;
  out->patch = c->patch;
  out->build = c->build;
  out->cuda_driver = c->cuda_driver;
  out->plugins = c->plugins;
  out->device = c->device;
  out->device_name = t->device_name;
  out->cc_major = c->cc_major;
  out->cc_minor = c->cc_minor;
}

int jl_trt_mem_info(jl_trt *t, size_t *free_bytes, size_t *total_bytes, char *err, size_t errlen) {
  int rc = enter(t, "mem_info", err, errlen);
  if (rc != JL_TRT_OK) {
    return rc;
  }
  size_t used = 0;
  for (obj *o = t->live; o != NULL; o = o->next) {
    if (o->kind == OBJ_DEVICE) {
      used += o->size;
    }
  }
  *total_bytes = t->config.total_memory;
  *free_bytes = used < t->config.total_memory ? t->config.total_memory - used : 0;
  return leave(t, JL_TRT_OK);
}

void jl_trt_set_logger(jl_trt *t, int min_severity, jl_trt_log_fn fn, void *ctx) {
  pthread_mutex_lock(&t->lock);
  t->log = fn;
  t->log_ctx = ctx;
  t->log_min = min_severity;
  pthread_mutex_unlock(&t->lock);
}

int jl_trt_sticky(const jl_trt *t) {
  jl_trt *m = (jl_trt *)t;
  pthread_mutex_lock(&m->lock);
  int s = m->sticky;
  pthread_mutex_unlock(&m->lock);
  return s;
}

void jl_trt_fake_fail(jl_trt *t, const char *call, int nth, int code, const char *message) {
  pthread_mutex_lock(&t->lock);
  if (t->n_fails < MAX_FAILS) {
    fail *f = &t->fails[t->n_fails++];
    snprintf(f->call, sizeof f->call, "%s", call);
    f->nth = nth < 1 ? 1 : nth;
    f->code = code;
    snprintf(f->message, sizeof f->message, "%s", message ? message : "injected");
  }
  pthread_mutex_unlock(&t->lock);
}

void jl_trt_fake_set_build(jl_trt *t, const char *tensor_lines, int layers) {
  pthread_mutex_lock(&t->lock);
  free(t->build_tensors);
  t->build_tensors = tensor_lines ? strdup(tensor_lines) : NULL;
  t->build_layers = layers;
  pthread_mutex_unlock(&t->lock);
}

void jl_trt_fake_get_stats(jl_trt *t, jl_trt_fake_stats *out) {
  pthread_mutex_lock(&t->lock);
  *out = t->stats;
  int64_t *counts[] = {&out->device_allocs, &out->host_allocs, &out->streams, &out->events, &out->graphs,
                       &out->graph_execs,   &out->engines,     &out->contexts, &out->builds};
  for (size_t i = 0; i < sizeof counts / sizeof counts[0]; i++) {
    *counts[i] = 0;
  }
  for (obj *o = t->live; o != NULL; o = o->next) {
    (*counts[o->kind])++;
  }
  pthread_mutex_unlock(&t->lock);
}

const char *jl_trt_fake_last_misuse(jl_trt *t) {
  return t->misuse;
}

// --- memory ------------------------------------------------------------------------

static int alloc(jl_trt *t, const char *call, int kind, size_t size, void **out, char *err, size_t errlen) {
  int rc = enter(t, call, err, errlen);
  if (rc != JL_TRT_OK) {
    return rc;
  }
  const char *cu = kind == OBJ_DEVICE ? "cuMemAlloc" : "cuMemHostAlloc";
  if ((rc = unsafe(t, cu, err, errlen)) != JL_TRT_OK) {
    return leave(t, rc);
  }
  if (size == 0) {
    return leave(t, say(err, errlen, JL_TRT_CUDA_ERROR, "%s: CUDA_ERROR_INVALID_VALUE: invalid argument", cu));
  }
  void *p = malloc(size);
  if (p == NULL) {
    return leave(t, say(err, errlen, JL_TRT_CUDA_ERROR, "%s: CUDA_ERROR_OUT_OF_MEMORY: out of memory", cu));
  }
  memset(p, 0xff, size);
  track(t, kind, p, size);
  *out = p;
  return leave(t, JL_TRT_OK);
}

static void release(jl_trt *t, int kind, void *p) {
  if (t == NULL || p == NULL) {
    return;
  }
  pthread_mutex_lock(&t->lock);
  if (unsafe(t, kind == OBJ_DEVICE ? "cuMemFree" : "cuMemFreeHost", NULL, 0) != JL_TRT_OK) {
    misuse(t, "memory freed on the thread that is capturing");
  }
  if (untrack(t, kind, p)) {
    free(p);
  } else {
    misuse(t, "freeing %s memory that is not live", kind == OBJ_DEVICE ? "device" : "host");
  }
  pthread_mutex_unlock(&t->lock);
}

int jl_trt_mem_alloc(jl_trt *t, size_t size, jl_trt_dptr *out, char *err, size_t errlen) {
  void *p = NULL;
  int rc = alloc(t, "mem_alloc", OBJ_DEVICE, size, &p, err, errlen);
  *out = (jl_trt_dptr)(uintptr_t)p;
  return rc;
}

void jl_trt_mem_free(jl_trt *t, jl_trt_dptr ptr) {
  release(t, OBJ_DEVICE, (void *)(uintptr_t)ptr);
}

int jl_trt_host_alloc(jl_trt *t, size_t size, void **out, char *err, size_t errlen) {
  *out = NULL;
  return alloc(t, "host_alloc", OBJ_HOST, size, out, err, errlen);
}

void jl_trt_host_free(jl_trt *t, void *ptr) {
  release(t, OBJ_HOST, ptr);
}

static int queue(jl_trt *t, const char *call, jl_trt_stream *s, op o, char *err, size_t errlen) {
  int rc = enter(t, call, err, errlen);
  if (rc != JL_TRT_OK) {
    return rc;
  }
  return leave(t, submit(t, s, o, err, errlen));
}

int jl_trt_copy_h2d(jl_trt *t, jl_trt_dptr dst, const void *src, size_t size, jl_trt_stream *stream, char *err,
                    size_t errlen) {
  op o = {OP_H2D, (uintptr_t)dst, (uintptr_t)src, size, 0, NULL, NULL};
  return queue(t, "copy_h2d", stream, o, err, errlen);
}

int jl_trt_copy_d2h(jl_trt *t, void *dst, jl_trt_dptr src, size_t size, jl_trt_stream *stream, char *err,
                    size_t errlen) {
  op o = {OP_D2H, (uintptr_t)dst, (uintptr_t)src, size, 0, NULL, NULL};
  return queue(t, "copy_d2h", stream, o, err, errlen);
}

int jl_trt_copy_d2d(jl_trt *t, jl_trt_dptr dst, jl_trt_dptr src, size_t size, jl_trt_stream *stream, char *err,
                    size_t errlen) {
  op o = {OP_D2D, (uintptr_t)dst, (uintptr_t)src, size, 0, NULL, NULL};
  return queue(t, "copy_d2d", stream, o, err, errlen);
}

int jl_trt_memset(jl_trt *t, jl_trt_dptr dst, uint8_t value, size_t size, jl_trt_stream *stream, char *err,
                  size_t errlen) {
  op o = {OP_MEMSET, (uintptr_t)dst, 0, size, value, NULL, NULL};
  return queue(t, "memset", stream, o, err, errlen);
}

// --- streams and events ----------------------------------------------------------

int jl_trt_stream_create(jl_trt *t, jl_trt_stream **out, char *err, size_t errlen) {
  *out = NULL;
  int rc = enter(t, "stream_create", err, errlen);
  if (rc != JL_TRT_OK) {
    return rc;
  }
  jl_trt_stream *s = calloc(1, sizeof(jl_trt_stream));
  if (s == NULL) {
    return leave(t, say(err, errlen, JL_TRT_CUDA_ERROR, "cuStreamCreate: CUDA_ERROR_OUT_OF_MEMORY: out of memory"));
  }
  s->trt = t;
  track(t, OBJ_STREAM, s, 0);
  *out = s;
  return leave(t, JL_TRT_OK);
}

int jl_trt_stream_sync(jl_trt *t, jl_trt_stream *s, char *err, size_t errlen) {
  int rc = enter(t, "stream_sync", err, errlen);
  if (rc != JL_TRT_OK) {
    return rc;
  }
  if ((rc = unsafe(t, "cuStreamSynchronize", err, errlen)) != JL_TRT_OK) {
    return leave(t, rc);
  }
  if (find(t, OBJ_STREAM, s) == NULL) {
    misuse(t, "syncing a stream that is not live");
    return leave(t, say(err, errlen, JL_TRT_CUDA_ERROR, "cuStreamSynchronize: CUDA_ERROR_INVALID_HANDLE: invalid resource handle"));
  }
  return leave(t, JL_TRT_OK);
}

void jl_trt_stream_destroy(jl_trt *t, jl_trt_stream *s) {
  if (t == NULL || s == NULL) {
    return;
  }
  pthread_mutex_lock(&t->lock);
  if (find(t, OBJ_STREAM, s) == NULL) {
    misuse(t, "destroying a stream that is not live");
  } else {
    if (s->capturing) {
      misuse(t, "destroying a stream that is capturing");
      if (t->capturing == s) {
        t->capturing = NULL;
      }
    }
    free_ops(&s->ops);
    untrack(t, OBJ_STREAM, s);
    free(s);
  }
  pthread_mutex_unlock(&t->lock);
}

int jl_trt_event_create(jl_trt *t, unsigned flags, jl_trt_event **out, char *err, size_t errlen) {
  *out = NULL;
  int rc = enter(t, "event_create", err, errlen);
  if (rc != JL_TRT_OK) {
    return rc;
  }
  jl_trt_event *e = calloc(1, sizeof(jl_trt_event));
  if (e == NULL) {
    return leave(t, say(err, errlen, JL_TRT_CUDA_ERROR, "cuEventCreate: CUDA_ERROR_OUT_OF_MEMORY: out of memory"));
  }
  e->flags = flags;
  track(t, OBJ_EVENT, e, 0);
  *out = e;
  return leave(t, JL_TRT_OK);
}

int jl_trt_event_record(jl_trt *t, jl_trt_event *event, jl_trt_stream *s, unsigned flags, char *err, size_t errlen) {
  int rc = enter(t, "event_record", err, errlen);
  if (rc != JL_TRT_OK) {
    return rc;
  }
  if (find(t, OBJ_EVENT, event) == NULL) {
    misuse(t, "recording an event that is not live");
    return leave(t, say(err, errlen, JL_TRT_CUDA_ERROR, "cuEventRecordWithFlags: CUDA_ERROR_INVALID_HANDLE: invalid resource handle"));
  }
  if (find(t, OBJ_STREAM, s) != NULL && s->capturing && !(flags & JL_TRT_RECORD_EXTERNAL)) {
    // a dependency inside the graph, not something the host can wait on
    if (!s->invalidated) {
      event->internal = 1;
    }
    return leave(t, JL_TRT_OK);
  }
  op o = {OP_RECORD, 0, 0, 0, 0, NULL, event};
  return leave(t, submit(t, s, o, err, errlen));
}

static int host_wait(jl_trt *t, const char *call, jl_trt_event *event, char *err, size_t errlen) {
  int rc = unsafe(t, call, err, errlen);
  if (rc != JL_TRT_OK) {
    return rc;
  }
  if (find(t, OBJ_EVENT, event) == NULL) {
    misuse(t, "waiting on an event that is not live");
    return say(err, errlen, JL_TRT_CUDA_ERROR, "%s: CUDA_ERROR_INVALID_HANDLE: invalid resource handle", call);
  }
  if (event->internal) {
    misuse(t, "the host waited on an event recorded inside a capture without JL_TRT_RECORD_EXTERNAL");
    return say(err, errlen, JL_TRT_CUDA_ERROR,
               "%s: CUDA_ERROR_INVALID_VALUE: invalid argument (fake: recorded inside a capture without "
               "JL_TRT_RECORD_EXTERNAL)",
               call);
  }
  return JL_TRT_OK;
}

int jl_trt_event_sync(jl_trt *t, jl_trt_event *event, char *err, size_t errlen) {
  int rc = enter(t, "event_sync", err, errlen);
  if (rc != JL_TRT_OK) {
    return rc;
  }
  return leave(t, host_wait(t, "cuEventSynchronize", event, err, errlen));
}

int jl_trt_event_query(jl_trt *t, jl_trt_event *event, char *err, size_t errlen) {
  int rc = enter(t, "event_query", err, errlen);
  if (rc != JL_TRT_OK) {
    return rc;
  }
  // everything here has finished by the time it returns
  return leave(t, host_wait(t, "cuEventQuery", event, err, errlen));
}

int jl_trt_event_elapsed(jl_trt *t, jl_trt_event *start, jl_trt_event *end, float *ms, char *err, size_t errlen) {
  int rc = enter(t, "event_elapsed", err, errlen);
  if (rc != JL_TRT_OK) {
    return rc;
  }
  if (find(t, OBJ_EVENT, start) == NULL || find(t, OBJ_EVENT, end) == NULL ||
      ((start->flags | end->flags) & JL_TRT_EVENT_DISABLE_TIMING) || !start->recorded || !end->recorded) {
    return leave(t, say(err, errlen, JL_TRT_CUDA_ERROR, "cuEventElapsedTime: CUDA_ERROR_INVALID_HANDLE: invalid resource handle"));
  }
  *ms = (float)(end->stamp - start->stamp);
  return leave(t, JL_TRT_OK);
}

void jl_trt_event_destroy(jl_trt *t, jl_trt_event *event) {
  if (t == NULL || event == NULL) {
    return;
  }
  pthread_mutex_lock(&t->lock);
  if (untrack(t, OBJ_EVENT, event)) {
    free(event);
  } else {
    misuse(t, "destroying an event that is not live");
  }
  pthread_mutex_unlock(&t->lock);
}

// --- graphs ------------------------------------------------------------------------

int jl_trt_capture_begin(jl_trt *t, jl_trt_stream *s, char *err, size_t errlen) {
  int rc = enter(t, "capture_begin", err, errlen);
  if (rc != JL_TRT_OK) {
    return rc;
  }
  if (find(t, OBJ_STREAM, s) == NULL) {
    return leave(t, say(err, errlen, JL_TRT_CUDA_ERROR, "cuStreamBeginCapture: CUDA_ERROR_INVALID_HANDLE: invalid resource handle"));
  }
  if (s->capturing || t->capturing != NULL) {
    misuse(t, "a capture began while another was running");
    return leave(t, say(err, errlen, JL_TRT_CUDA_ERROR,
                        "cuStreamBeginCapture: CUDA_ERROR_ILLEGAL_STATE: the operation cannot be performed in the present state"));
  }
  s->capturing = 1;
  s->invalidated = 0;
  free_ops(&s->ops);
  t->capturing = s;
  t->capture_thread = pthread_self();
  return leave(t, JL_TRT_OK);
}

int jl_trt_capture_end(jl_trt *t, jl_trt_stream *s, jl_trt_graph **out, char *err, size_t errlen) {
  *out = NULL;
  int rc = enter(t, "capture_end", err, errlen);
  if (rc != JL_TRT_OK) {
    return rc;
  }
  if (find(t, OBJ_STREAM, s) == NULL || !s->capturing) {
    return leave(t, say(err, errlen, JL_TRT_CUDA_ERROR,
                        "cuStreamEndCapture: CUDA_ERROR_ILLEGAL_STATE: the operation cannot be performed in the present state"));
  }
  if (!pthread_equal(t->capture_thread, pthread_self())) {
    return leave(t, say(err, errlen, JL_TRT_CUDA_ERROR,
                        "cuStreamEndCapture: CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD: attempt to terminate a thread-local "
                        "capture sequence from another thread"));
  }
  s->capturing = 0;
  t->capturing = NULL;
  if (s->invalidated) {
    free_ops(&s->ops);
    return leave(t, say(err, errlen, JL_TRT_CUDA_ERROR,
                        "cuStreamEndCapture: CUDA_ERROR_STREAM_CAPTURE_INVALIDATED: operation failed due to a previous "
                        "error during capture"));
  }
  jl_trt_graph *g = calloc(1, sizeof(jl_trt_graph));
  if (g == NULL) {
    free_ops(&s->ops);
    return leave(t, say(err, errlen, JL_TRT_CUDA_ERROR, "cuStreamEndCapture: CUDA_ERROR_OUT_OF_MEMORY: out of memory"));
  }
  g->ops = s->ops;
  s->ops.ops = NULL;
  s->ops.n = s->ops.cap = 0;
  t->stats.captured_ops = g->ops.n;
  track(t, OBJ_GRAPH, g, 0);
  *out = g;
  return leave(t, JL_TRT_OK);
}

int jl_trt_graph_instantiate(jl_trt *t, jl_trt_graph *graph, jl_trt_graph_exec **out, char *err, size_t errlen) {
  *out = NULL;
  int rc = enter(t, "graph_instantiate", err, errlen);
  if (rc != JL_TRT_OK) {
    return rc;
  }
  if (find(t, OBJ_GRAPH, graph) == NULL) {
    return leave(t, say(err, errlen, JL_TRT_CUDA_ERROR, "cuGraphInstantiateWithFlags: CUDA_ERROR_INVALID_VALUE: invalid argument"));
  }
  jl_trt_graph_exec *x = calloc(1, sizeof(jl_trt_graph_exec));
  if (x == NULL || !copy_ops(&x->ops, &graph->ops)) {
    free(x);
    return leave(t, say(err, errlen, JL_TRT_CUDA_ERROR, "cuGraphInstantiateWithFlags: CUDA_ERROR_OUT_OF_MEMORY: out of memory"));
  }
  track(t, OBJ_EXEC, x, 0);
  *out = x;
  return leave(t, JL_TRT_OK);
}

int jl_trt_graph_launch(jl_trt *t, jl_trt_graph_exec *exec, jl_trt_stream *s, char *err, size_t errlen) {
  int rc = enter(t, "graph_launch", err, errlen);
  if (rc != JL_TRT_OK) {
    return rc;
  }
  if (find(t, OBJ_EXEC, exec) == NULL || find(t, OBJ_STREAM, s) == NULL) {
    misuse(t, "launching a graph or onto a stream that is not live");
    return leave(t, say(err, errlen, JL_TRT_CUDA_ERROR, "cuGraphLaunch: CUDA_ERROR_INVALID_HANDLE: invalid resource handle"));
  }
  t->stats.graph_launches++;
  for (size_t i = 0; i < exec->ops.n; i++) {
    rc = s->capturing ? submit(t, s, exec->ops.ops[i], err, errlen) : execute(t, &exec->ops.ops[i], 1, err, errlen);
    if (rc != JL_TRT_OK) {
      return leave(t, rc);
    }
  }
  return leave(t, JL_TRT_OK);
}

void jl_trt_graph_destroy(jl_trt *t, jl_trt_graph *graph) {
  if (t == NULL || graph == NULL) {
    return;
  }
  pthread_mutex_lock(&t->lock);
  if (untrack(t, OBJ_GRAPH, graph)) {
    free_ops(&graph->ops);
    free(graph);
  } else {
    misuse(t, "destroying a graph that is not live");
  }
  pthread_mutex_unlock(&t->lock);
}

void jl_trt_graph_exec_destroy(jl_trt *t, jl_trt_graph_exec *exec) {
  if (t == NULL || exec == NULL) {
    return;
  }
  pthread_mutex_lock(&t->lock);
  if (untrack(t, OBJ_EXEC, exec)) {
    free_ops(&exec->ops);
    free(exec);
  } else {
    misuse(t, "destroying a graph exec that is not live");
  }
  pthread_mutex_unlock(&t->lock);
}

// --- runtime -------------------------------------------------------------------------

static int plan_error(jl_trt *t, char *err, size_t errlen, const char *fmt, ...) {
  char line[256];
  va_list args;
  va_start(args, fmt);
  vsnprintf(line, sizeof line, fmt, args);
  va_end(args);
  logf_(t, JL_TRT_LOG_ERROR, "%s", line);
  return say(err, errlen, JL_TRT_ERROR, "%s", line);
}

static int parse_plan(jl_trt *t, char *text, jl_trt_engine *e, char *err, size_t errlen) {
  char *save = NULL;
  int line_no = 0, magic = 0;
  for (char *line = strtok_r(text, "\n", &save); line != NULL; line = strtok_r(NULL, "\n", &save)) {
    line_no++;
    char *words[4 + JL_TRT_MAX_DIMS + 2];
    int n = 0;
    char *wsave = NULL;
    for (char *w = strtok_r(line, " \t\r", &wsave); w != NULL; w = strtok_r(NULL, " \t\r", &wsave)) {
      if (n == (int)(sizeof words / sizeof words[0])) {
        return plan_error(t, err, errlen, "fake: plan line %d has too many words", line_no);
      }
      words[n++] = w;
    }
    if (n == 0 || words[0][0] == '#') {
      continue;
    }
    if (!magic) {
      if (n != 2 || strcmp(words[0], "jl_trt_fake_plan") != 0 || strcmp(words[1], "1") != 0) {
        return plan_error(t, err, errlen, "fake: not a fake plan");
      }
      magic = 1;
    } else if (strcmp(words[0], "built") == 0 && n == 2) {
      char have[64];
      snprintf(have, sizeof have, "%d.%d.%d.%d", t->config.major, t->config.minor, t->config.patch, t->config.build);
      if (strcmp(words[1], have) != 0) {
        return plan_error(t, err, errlen, "fake: the plan was built by TensorRT %s, this is %s; rebuild it", words[1], have);
      }
    } else if (strcmp(words[0], "settings") == 0) {
      continue;
    } else if ((strcmp(words[0], "input") == 0 || strcmp(words[0], "output") == 0) && n >= 4) {
      if (e->n == MAX_TENSORS) {
        return plan_error(t, err, errlen, "fake: plan line %d: too many tensors", line_no);
      }
      tensor *x = &e->tensors[e->n];
      memset(x, 0, sizeof *x);
      x->is_input = words[0][0] == 'i';
      x->from = -1;
      snprintf(x->name, sizeof x->name, "%s", words[1]);
      int known = 0;
      for (size_t i = 0; i < sizeof type_names / sizeof type_names[0]; i++) {
        if (strcmp(words[2], type_names[i].name) == 0) {
          x->type = type_names[i].type;
          known = 1;
        }
      }
      if (!known) {
        return plan_error(t, err, errlen, "fake: plan line %d: unknown type %s", line_no, words[2]);
      }
      int w = 3;
      for (; w < n && strcmp(words[w], "from") != 0; w++) {
        if (x->rank == JL_TRT_MAX_DIMS) {
          return plan_error(t, err, errlen, "fake: plan line %d: too many dims", line_no);
        }
        x->dims[x->rank++] = strtoll(words[w], NULL, 10);
      }
      if (w < n) {
        if (x->is_input || w + 2 != n) {
          return plan_error(t, err, errlen, "fake: plan line %d: only an output is fed `from` one input", line_no);
        }
        for (int i = 0; i < e->n; i++) {
          if (e->tensors[i].is_input && strcmp(e->tensors[i].name, words[w + 1]) == 0) {
            x->from = i;
          }
        }
        if (x->from < 0) {
          return plan_error(t, err, errlen, "fake: plan line %d: no input %s before it", line_no, words[w + 1]);
        }
      }
      e->n++;
    } else {
      return plan_error(t, err, errlen, "fake: plan line %d is not understood", line_no);
    }
  }
  return magic ? JL_TRT_OK : plan_error(t, err, errlen, "fake: not a fake plan");
}

int jl_trt_engine_deserialize(jl_trt *t, const void *plan, size_t size, jl_trt_engine **out, char *err,
                              size_t errlen) {
  *out = NULL;
  int rc = enter(t, "engine_deserialize", err, errlen);
  if (rc != JL_TRT_OK) {
    return rc;
  }
  if ((rc = unsafe(t, "cuMemAlloc", err, errlen)) != JL_TRT_OK) {
    return leave(t, rc);
  }
  char *text = malloc(size + 1);
  jl_trt_engine *e = calloc(1, sizeof(jl_trt_engine));
  if (text == NULL || e == NULL) {
    free(text);
    free(e);
    return leave(t, say(err, errlen, JL_TRT_ERROR, "out of memory"));
  }
  if (size > 0) {
    memcpy(text, plan, size);
  }
  text[size] = '\0';
  e->trt = t;
  rc = parse_plan(t, text, e, err, errlen);
  free(text);
  if (rc != JL_TRT_OK) {
    free(e);
    return leave(t, rc);
  }
  track(t, OBJ_ENGINE, e, 0);
  *out = e;
  return leave(t, JL_TRT_OK);
}

void jl_trt_engine_destroy(jl_trt_engine *e) {
  if (e == NULL) {
    return;
  }
  jl_trt *t = e->trt;
  pthread_mutex_lock(&t->lock);
  for (obj *o = t->live; o != NULL; o = o->next) {
    if (o->kind == OBJ_CONTEXT && ((jl_trt_context *)o->ptr)->engine == e) {
      misuse(t, "an engine destroyed before its execution context");
      break;
    }
  }
  if (untrack(t, OBJ_ENGINE, e)) {
    free(e);
  } else {
    misuse(t, "destroying an engine that is not live");
  }
  pthread_mutex_unlock(&t->lock);
}

int jl_trt_engine_io_count(const jl_trt_engine *e) {
  return e->n;
}

int jl_trt_engine_io(const jl_trt_engine *e, int index, const char **name, int *is_input, int *type, int64_t *dims,
                     int *rank, char *err, size_t errlen) {
  if (index < 0 || index >= e->n) {
    return say(err, errlen, JL_TRT_ERROR, "no IO tensor %d; the engine has %d", index, e->n);
  }
  const tensor *x = &e->tensors[index];
  *name = x->name;
  *is_input = x->is_input;
  *type = x->type;
  *rank = x->rank;
  memcpy(dims, x->dims, (size_t)x->rank * sizeof(int64_t));
  return JL_TRT_OK;
}

int jl_trt_context_create(jl_trt_engine *e, jl_trt_context **out, char *err, size_t errlen) {
  *out = NULL;
  jl_trt *t = e->trt;
  int rc = enter(t, "context_create", err, errlen);
  if (rc != JL_TRT_OK) {
    return rc;
  }
  if ((rc = unsafe(t, "cuMemAlloc", err, errlen)) != JL_TRT_OK) {
    return leave(t, rc);
  }
  jl_trt_context *c = calloc(1, sizeof(jl_trt_context));
  if (c == NULL) {
    return leave(t, say(err, errlen, JL_TRT_ERROR, "out of memory"));
  }
  c->engine = e;
  track(t, OBJ_CONTEXT, c, 0);
  *out = c;
  return leave(t, JL_TRT_OK);
}

void jl_trt_context_destroy(jl_trt_context *c) {
  if (c == NULL) {
    return;
  }
  jl_trt *t = c->engine->trt;
  pthread_mutex_lock(&t->lock);
  if (untrack(t, OBJ_CONTEXT, c)) {
    free(c);
  } else {
    misuse(t, "destroying an execution context that is not live");
  }
  pthread_mutex_unlock(&t->lock);
}

int jl_trt_context_set_address(jl_trt_context *c, const char *name, jl_trt_dptr address, char *err, size_t errlen) {
  jl_trt *t = c->engine->trt;
  int rc = enter(t, "context_set_address", err, errlen);
  if (rc != JL_TRT_OK) {
    return rc;
  }
  for (int i = 0; i < c->engine->n; i++) {
    if (strcmp(c->engine->tensors[i].name, name) == 0) {
      c->address[i] = address;
      return leave(t, JL_TRT_OK);
    }
  }
  return leave(t, plan_error(t, err, errlen, "IExecutionContext::setTensorAddress: no IO tensor named %s", name));
}

int jl_trt_context_enqueue(jl_trt_context *c, jl_trt_stream *s, char *err, size_t errlen) {
  jl_trt *t = c->engine->trt;
  int rc = enter(t, "context_enqueue", err, errlen);
  if (rc != JL_TRT_OK) {
    return rc;
  }
  if ((rc = check_bound(t, c, err, errlen)) != JL_TRT_OK) {
    if (find(t, OBJ_STREAM, s) != NULL && s->capturing) {
      s->invalidated = 1;
    }
    return leave(t, rc);
  }
  if (find(t, OBJ_STREAM, s) != NULL && s->capturing && !c->ran) {
    s->invalidated = 1;
    return leave(t, plan_error(t, err, errlen, "fake: enqueue once before capturing, as TensorRT requires"));
  }
  op o = {OP_ENQUEUE, 0, 0, 0, 0, c, NULL};
  int direct = find(t, OBJ_STREAM, s) != NULL && !s->capturing;
  rc = submit(t, s, o, err, errlen);
  if (rc == JL_TRT_OK && direct) {
    t->stats.direct_enqueues++;
  }
  return leave(t, rc);
}

// --- builder -------------------------------------------------------------------------

int jl_trt_build_create(jl_trt *t, jl_trt_build **out, char *err, size_t errlen) {
  *out = NULL;
  int rc = enter(t, "build_create", err, errlen);
  if (rc != JL_TRT_OK) {
    return rc;
  }
  if (t->config.no_parser) {
    return leave(t, say(err, errlen, JL_TRT_UNAVAILABLE, "libnvonnxparser.so.%d: cannot open shared object file", t->config.major));
  }
  jl_trt_build *b = calloc(1, sizeof(jl_trt_build));
  if (b == NULL) {
    return leave(t, say(err, errlen, JL_TRT_ERROR, "out of memory"));
  }
  b->trt = t;
  b->optimization_level = 3;
  b->cache_builds = -1;
  track(t, OBJ_BUILD, b, 0);
  *out = b;
  return leave(t, JL_TRT_OK);
}

void jl_trt_build_destroy(jl_trt_build *b) {
  if (b == NULL) {
    return;
  }
  jl_trt *t = b->trt;
  pthread_mutex_lock(&t->lock);
  if (untrack(t, OBJ_BUILD, b)) {
    free(b);
  } else {
    misuse(t, "destroying a build that is not live");
  }
  pthread_mutex_unlock(&t->lock);
}

int jl_trt_build_parse(jl_trt_build *b, const char *onnx_path, char *err, size_t errlen) {
  jl_trt *t = b->trt;
  int rc = enter(t, "build_parse", err, errlen);
  if (rc != JL_TRT_OK) {
    return rc;
  }
  FILE *f = fopen(onnx_path, "rb");
  if (f == NULL) {
    return leave(t, say(err, errlen, JL_TRT_ERROR, "(parseFromFile): MODEL_DESERIALIZE_FAILED: fake: cannot open %s", onnx_path));
  }
  fclose(f);
  b->parsed = 1;
  return leave(t, JL_TRT_OK);
}

int jl_trt_build_layers(const jl_trt_build *b) {
  return b->parsed ? b->trt->build_layers : 0;
}

int jl_trt_build_set_fp16(jl_trt_build *b, char *err, size_t errlen) {
  jl_trt *t = b->trt;
  int rc = enter(t, "build_set_fp16", err, errlen);
  if (rc != JL_TRT_OK) {
    return rc;
  }
  if (t->config.strongly_typed) {
    return leave(t, say(err, errlen, JL_TRT_ERROR, "TensorRT %d has no FP16 flag; precision follows the ONNX", t->config.major));
  }
  b->fp16 = 1;
  return leave(t, JL_TRT_OK);
}

void jl_trt_build_set_optimization_level(jl_trt_build *b, int level) {
  b->optimization_level = level;
}

void jl_trt_build_set_workspace(jl_trt_build *b, size_t bytes) {
  b->workspace = bytes;
}

void jl_trt_build_set_progress(jl_trt_build *b, jl_trt_progress_fn fn, void *ctx) {
  b->progress = fn;
  b->progress_ctx = ctx;
}

int jl_trt_build_set_timing_cache(jl_trt_build *b, const void *data, size_t size, char *err, size_t errlen) {
  jl_trt *t = b->trt;
  int rc = enter(t, "build_set_timing_cache", err, errlen);
  if (rc != JL_TRT_OK) {
    return rc;
  }
  b->cache_builds = -1;
  if (data == NULL || size == 0) {
    b->cache_builds = 0;
    return leave(t, JL_TRT_OK);
  }
  char text[128];
  size_t n = size < sizeof text - 1 ? size : sizeof text - 1;
  memcpy(text, data, n);
  text[n] = '\0';
  int major, minor, patch, build, builds;
  if (sscanf(text, "jl_trt_fake_timing %d.%d.%d.%d %d", &major, &minor, &patch, &build, &builds) != 5) {
    return leave(t, plan_error(t, err, errlen, "fake: the timing cache is truncated or not one"));
  }
  if (major != t->config.major || minor != t->config.minor || patch != t->config.patch || build != t->config.build) {
    return leave(t, plan_error(t, err, errlen, "fake: the timing cache is from TensorRT %d.%d.%d.%d", major, minor, patch, build));
  }
  b->cache_builds = builds;
  return leave(t, JL_TRT_OK);
}

static void progress(jl_trt_build *b, int event, const char *phase, const char *parent, int value) {
  if (b->progress != NULL) {
    b->progress(b->progress_ctx, event, phase, parent, value);
  }
}

static int write_file(const char *path, const char *text, char *err, size_t errlen) {
  FILE *f = fopen(path, "wb");
  if (f == NULL) {
    return say(err, errlen, JL_TRT_ERROR, "cannot write %s", path);
  }
  size_t n = strlen(text);
  int ok = fwrite(text, 1, n, f) == n;
  ok = fclose(f) == 0 && ok;
  return ok ? JL_TRT_OK : say(err, errlen, JL_TRT_ERROR, "cannot write %s", path);
}

int jl_trt_build_write_plan(jl_trt_build *b, const char *path, char *err, size_t errlen) {
  jl_trt *t = b->trt;
  int rc = enter(t, "build_write_plan", err, errlen);
  if (rc != JL_TRT_OK) {
    return rc;
  }
  if (!b->parsed) {
    return leave(t, plan_error(t, err, errlen, "fake: nothing parsed to build"));
  }
  logf_(t, JL_TRT_LOG_WARNING, "fake: building a plan");
  int layers = t->build_layers > 0 ? t->build_layers : 1;
  progress(b, JL_TRT_PHASE_START, "fake build", NULL, layers);
  for (int i = 0; i < layers; i++) {
    progress(b, JL_TRT_PHASE_START, "fake tactics", "fake build", 2);
    progress(b, JL_TRT_PHASE_STEP, "fake tactics", NULL, 0);
    progress(b, JL_TRT_PHASE_STEP, "fake tactics", NULL, 1);
    progress(b, JL_TRT_PHASE_FINISH, "fake tactics", NULL, 0);
    progress(b, JL_TRT_PHASE_STEP, "fake build", NULL, i);
  }
  progress(b, JL_TRT_PHASE_FINISH, "fake build", NULL, 0);
  const char *tensors = t->build_tensors ? t->build_tensors : default_tensors;
  const char *cache = b->cache_builds < 0 ? "none" : b->cache_builds == 0 ? "cold" : "warm";
  size_t cap = strlen(tensors) + 256;
  char *text = malloc(cap);
  if (text == NULL) {
    return leave(t, say(err, errlen, JL_TRT_ERROR, "out of memory"));
  }
  snprintf(text, cap,
           "jl_trt_fake_plan 1\nbuilt %d.%d.%d.%d\nsettings fp16=%d optimization_level=%d workspace=%zu timing_cache=%s\n%s",
           t->config.major, t->config.minor, t->config.patch, t->config.build, b->fp16, b->optimization_level,
           b->workspace, cache, tensors);
  rc = write_file(path, text, err, errlen);
  free(text);
  if (rc == JL_TRT_OK && b->cache_builds >= 0) {
    b->cache_builds++;
  }
  return leave(t, rc);
}

int jl_trt_build_write_timing_cache(jl_trt_build *b, const char *path, char *err, size_t errlen) {
  jl_trt *t = b->trt;
  int rc = enter(t, "build_write_timing_cache", err, errlen);
  if (rc != JL_TRT_OK) {
    return rc;
  }
  if (b->cache_builds < 0) {
    return leave(t, say(err, errlen, JL_TRT_ERROR, "no timing cache is attached"));
  }
  char text[128];
  snprintf(text, sizeof text, "jl_trt_fake_timing %d.%d.%d.%d %d\n", t->config.major, t->config.minor, t->config.patch,
           t->config.build, b->cache_builds);
  return leave(t, write_file(path, text, err, errlen));
}
