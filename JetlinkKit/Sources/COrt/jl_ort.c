#include "jl_ort.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <onnxruntime/onnxruntime_c_api.h>

#ifdef JL_ORT_DLOPEN
#include <dlfcn.h>
#endif

struct jl_env {
  OrtEnv *env;
};

struct jl_session {
  OrtSession *session;
};

struct jl_binding {
  OrtSession *session;
  OrtIoBinding *binding;
  OrtValue **values;
  size_t n_values;
};

#ifdef JL_ORT_DLOPEN
// NULL when the library or its entry point is missing; jl_env_create says why.
static const OrtApiBase *base(void) {
  static const OrtApiBase *cached;
  if (cached == NULL) {
    void *library = dlopen("libonnxruntime.so", RTLD_NOW | RTLD_LOCAL);
    if (library != NULL) {
      const OrtApiBase *(*get)(void) = (const OrtApiBase *(*)(void))dlsym(library, "OrtGetApiBase");
      if (get != NULL) {
        cached = get();
      }
    }
  }
  return cached;
}
#else
static const OrtApiBase *base(void) {
  return OrtGetApiBase();
}
#endif

static const OrtApi *api(void) {
  static const OrtApi *cached;
  if (cached == NULL && base() != NULL) {
    cached = base()->GetApi(ORT_API_VERSION);
  }
  return cached;
}

static char *copy(const char *s) {
  size_t n = strlen(s) + 1;
  char *out = malloc(n);
  if (out != NULL) {
    memcpy(out, s, n);
  }
  return out;
}

// NULL for no status, else the message, and the status released.
static char *take(OrtStatus *status) {
  if (status == NULL) {
    return NULL;
  }
  char *message = copy(api()->GetErrorMessage(status));
  api()->ReleaseStatus(status);
  return message != NULL ? message : copy("onnxruntime failed");
}

#define TRY(call)                    \
  do {                               \
    char *error_ = take(call);       \
    if (error_ != NULL) {            \
      return error_;                 \
    }                                \
  } while (0)


// The runtime's own words, kept where a report can read them back. QNN's reason
// for taking no node of a graph is a line under onnxruntime's logger -- a device
// it could not create, a Stub the backend will not have, a graph its converter
// would not take, a Skel the DSP would not load -- and an Android app's stderr
// has no reader, so a line said there is a line lost. The ring holds the last of
// them, oldest first, and jl_log_lines hands them over.
#define JL_LOG_LINES 64
#define JL_LOG_CHARS 480

static char jl_log_ring[JL_LOG_LINES][JL_LOG_CHARS];
static size_t jl_log_next;
static size_t jl_log_held;
static char jl_log_joined[JL_LOG_LINES * JL_LOG_CHARS];

static const char *jl_log_severity(OrtLoggingLevel severity) {
  switch (severity) {
    case ORT_LOGGING_LEVEL_VERBOSE:
      return "verbose";
    case ORT_LOGGING_LEVEL_INFO:
      return "info";
    case ORT_LOGGING_LEVEL_WARNING:
      return "warning";
    case ORT_LOGGING_LEVEL_ERROR:
      return "error";
    case ORT_LOGGING_LEVEL_FATAL:
      return "fatal";
    default:
      return "log";
  }
}

// onnxruntime's logger, which carries the execution provider's own lines and
// QNN's beneath them: this is the one place both write to.
static void ORT_API_CALL jl_log_callback(void *param, OrtLoggingLevel severity, const char *category, const char *logid,
                                         const char *code_location, const char *message) {
  (void)param;
  // The same file, and the same logid, on every line of a session: the message
  // is the part to keep.
  (void)code_location;
  (void)logid;
  char line[JL_LOG_CHARS];
  snprintf(line, sizeof(line), "%s %s: %s", jl_log_severity(severity), category != NULL ? category : "",
           message != NULL ? message : "");
  snprintf(jl_log_ring[jl_log_next], JL_LOG_CHARS, "%s", line);
  jl_log_next = (jl_log_next + 1) % JL_LOG_LINES;
  if (jl_log_held < JL_LOG_LINES) {
    jl_log_held++;
  }
#ifndef __ANDROID__
  // Where a terminal reads it, the line goes on as it did before the ring.
  fprintf(stderr, "%s\n", line);
#endif
}

const char *jl_version(void) {
  return base() != NULL ? base()->GetVersionString() : "";
}

// The lines above, oldest first, newline-separated, "" when the runtime has said
// nothing yet. Overwritten by the next call; the caller does not free it.
const char *jl_log_lines(void) {
  jl_log_joined[0] = '\0';
  size_t start = (jl_log_next + JL_LOG_LINES - jl_log_held) % JL_LOG_LINES;
  size_t offset = 0;
  for (size_t i = 0; i < jl_log_held; i++) {
    const char *line = jl_log_ring[(start + i) % JL_LOG_LINES];
    size_t length = strlen(line);
    if (offset + length + 2 >= sizeof(jl_log_joined)) {
      break;
    }
    memcpy(jl_log_joined + offset, line, length);
    offset += length;
    jl_log_joined[offset++] = '\n';
  }
  jl_log_joined[offset] = '\0';
  return jl_log_joined;
}

void jl_free(void *p) {
  free(p);
}

char *jl_env_create(int log_severity, jl_env **out) {
  *out = NULL;
#ifdef JL_ORT_DLOPEN
  if (base() == NULL) {
    const char *why = dlerror();
    return copy(why != NULL ? why : "cannot open libonnxruntime.so");
  }
#endif
  if (api() == NULL) {
    return copy("this onnxruntime does not provide the C API version jetlink was built against");
  }
  jl_env *env = calloc(1, sizeof(jl_env));
  if (env == NULL) {
    return copy("out of memory");
  }
  // The official Apple build carries Microsoft's events SDK and uploads to
  // mobile.events.data.microsoft.com (seen from the iPhone simulator). The
  // switch below is read when the environment is created, which is before
  // DisableTelemetryEvents can be called, and events logged in between were
  // still uploaded. A car has no business sending any.
  setenv("ORT_DISABLE_TELEMETRY", "1", 1);
  // The ring above is the environment's logger: everything onnxruntime, the
  // execution provider and QNN say reaches it, and the `npu:` report reads it.
  char *error = take(api()->CreateEnvWithCustomLogger(jl_log_callback, NULL, (OrtLoggingLevel)log_severity, "jetlink", &env->env));
  if (error != NULL) {
    free(env);
    return error;
  }
  free(take(api()->DisableTelemetryEvents(env->env)));
  *out = env;
  return NULL;
}

void jl_env_release(jl_env *env) {
  if (env == NULL) {
    return;
  }
  api()->ReleaseEnv(env->env);
  free(env);
}

char *jl_session_create(jl_env *env, const char *model_path, const char *provider,
                        const jl_option *options, size_t n_options,
                        const jl_option *config, size_t n_config, int threads,
                        jl_session **out) {
  *out = NULL;
  OrtSessionOptions *so = NULL;
  TRY(api()->CreateSessionOptions(&so));
  char *error = take(api()->SetSessionGraphOptimizationLevel(so, ORT_ENABLE_ALL));
  if (error == NULL) {
    // A QNN session keeps onnxruntime's own words: QNN's reason for taking no
    // node of a graph is one it says below an error -- a Stub it cannot open,
    // a Skel the DSP will not take, a backend that is not its Stub's build --
    // and the `npu:` report reads them out of the phone's log. The CoreML
    // partitioner is chatty, so an Apple session keeps errors alone.
    int severity = provider != NULL && strcmp(provider, "QNN") == 0 ? 1 : 3;
    error = take(api()->SetSessionLogSeverityLevel(so, severity));
  }
  // One thread each way, and none that spin between frames: the CPU's share
  // of an accelerator's session is a couple of nodes, and a pool spinning for
  // the next frame burns a phone's battery and fights the frame path for cores.
  if (error == NULL) {
    error = take(api()->SetIntraOpNumThreads(so, threads > 0 ? threads : 1));
  }
  if (error == NULL) {
    error = take(api()->SetInterOpNumThreads(so, 1));
  }
  if (error == NULL) {
    error = take(api()->AddSessionConfigEntry(so, "session.intra_op.allow_spinning", "0"));
  }
  if (error == NULL) {
    error = take(api()->AddSessionConfigEntry(so, "session.inter_op.allow_spinning", "0"));
  }
  for (size_t i = 0; error == NULL && i < n_config; i++) {
    error = take(api()->AddSessionConfigEntry(so, config[i].key, config[i].value));
  }
  if (error == NULL && provider != NULL) {
    const char **keys = calloc(n_options + 1, sizeof(char *));
    const char **values = calloc(n_options + 1, sizeof(char *));
    if (keys == NULL || values == NULL) {
      error = copy("out of memory");
    } else {
      for (size_t i = 0; i < n_options; i++) {
        keys[i] = options[i].key;
        values[i] = options[i].value;
      }
      error = take(api()->SessionOptionsAppendExecutionProvider(so, provider, keys, values, n_options));
    }
    free(keys);
    free(values);
  }
  jl_session *session = NULL;
  if (error == NULL) {
    session = calloc(1, sizeof(jl_session));
    if (session == NULL) {
      error = copy("out of memory");
    }
  }
  if (error == NULL) {
    error = take(api()->CreateSession(env->env, model_path, so, &session->session));
  }
  api()->ReleaseSessionOptions(so);
  if (error != NULL) {
    free(session);
    return error;
  }
  *out = session;
  return NULL;
}

void jl_session_release(jl_session *session) {
  if (session == NULL) {
    return;
  }
  api()->ReleaseSession(session->session);
  free(session);
}

size_t jl_session_io_count(const jl_session *session, int output) {
  size_t n = 0;
  OrtStatus *status = output ? api()->SessionGetOutputCount(session->session, &n)
                             : api()->SessionGetInputCount(session->session, &n);
  free(take(status));
  return n;
}

char *jl_session_io_info(const jl_session *session, int output, size_t index,
                         char *name, size_t name_cap, int *elem_type,
                         int64_t *dims, size_t dims_cap, size_t *rank) {
  OrtAllocator *allocator = NULL;
  TRY(api()->GetAllocatorWithDefaultOptions(&allocator));
  char *raw = NULL;
  TRY(output ? api()->SessionGetOutputName(session->session, index, allocator, &raw)
             : api()->SessionGetInputName(session->session, index, allocator, &raw));
  if (name_cap > 0) {
    strncpy(name, raw, name_cap - 1);
    name[name_cap - 1] = '\0';
  }
  free(take(api()->AllocatorFree(allocator, raw)));

  OrtTypeInfo *info = NULL;
  TRY(output ? api()->SessionGetOutputTypeInfo(session->session, index, &info)
             : api()->SessionGetInputTypeInfo(session->session, index, &info));
  const OrtTensorTypeAndShapeInfo *tensor = NULL;
  char *error = take(api()->CastTypeInfoToTensorInfo(info, &tensor));
  if (error == NULL && tensor == NULL) {
    error = copy("not a tensor");
  }
  ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  size_t n = 0;
  if (error == NULL) {
    error = take(api()->GetTensorElementType(tensor, &type));
  }
  if (error == NULL) {
    error = take(api()->GetDimensionsCount(tensor, &n));
  }
  if (error == NULL) {
    int64_t *all = calloc(n > 0 ? n : 1, sizeof(int64_t));
    if (all == NULL) {
      error = copy("out of memory");
    } else {
      error = take(api()->GetDimensions(tensor, all, n));
      for (size_t i = 0; error == NULL && i < n && i < dims_cap; i++) {
        dims[i] = all[i];
      }
      free(all);
    }
  }
  api()->ReleaseTypeInfo(info);
  if (error != NULL) {
    return error;
  }
  *elem_type = (int)type;
  *rank = n;
  return NULL;
}

void jl_binding_release(jl_binding *binding) {
  if (binding == NULL) {
    return;
  }
  if (binding->binding != NULL) {
    api()->ReleaseIoBinding(binding->binding);
  }
  for (size_t i = 0; i < binding->n_values; i++) {
    if (binding->values[i] != NULL) {
      api()->ReleaseValue(binding->values[i]);
    }
  }
  free(binding->values);
  free(binding);
}

static char *wrap(const OrtMemoryInfo *memory, const jl_tensor *t, OrtValue **out) {
  return take(api()->CreateTensorWithDataAsOrtValue(memory, t->data, t->nbytes, t->dims, t->rank,
                                                    (ONNXTensorElementDataType)t->elem_type, out));
}

char *jl_binding_create(jl_session *session, const jl_tensor *inputs, size_t n_inputs,
                        const jl_tensor *outputs, size_t n_outputs, jl_binding **out) {
  *out = NULL;
  jl_binding *binding = calloc(1, sizeof(jl_binding));
  if (binding == NULL) {
    return copy("out of memory");
  }
  binding->session = session->session;
  binding->values = calloc(n_inputs + n_outputs + 1, sizeof(OrtValue *));
  if (binding->values == NULL) {
    free(binding);
    return copy("out of memory");
  }
  OrtMemoryInfo *memory = NULL;
  char *error = take(api()->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory));
  if (error == NULL) {
    error = take(api()->CreateIoBinding(session->session, &binding->binding));
  }
  for (size_t i = 0; error == NULL && i < n_inputs; i++) {
    OrtValue **value = &binding->values[binding->n_values++];
    error = wrap(memory, &inputs[i], value);
    if (error == NULL) {
      error = take(api()->BindInput(binding->binding, inputs[i].name, *value));
    }
  }
  for (size_t i = 0; error == NULL && i < n_outputs; i++) {
    OrtValue **value = &binding->values[binding->n_values++];
    error = wrap(memory, &outputs[i], value);
    if (error == NULL) {
      error = take(api()->BindOutput(binding->binding, outputs[i].name, *value));
    }
  }
  if (memory != NULL) {
    api()->ReleaseMemoryInfo(memory);
  }
  if (error != NULL) {
    jl_binding_release(binding);
    return error;
  }
  *out = binding;
  return NULL;
}

char *jl_binding_run(jl_binding *binding) {
  return take(api()->RunWithBinding(binding->session, NULL, binding->binding));
}
