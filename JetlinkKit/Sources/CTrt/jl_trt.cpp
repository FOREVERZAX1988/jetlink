// jl_trt.h over TensorRT and the CUDA driver API, both opened at run time.
//
// From TensorRT only the extern "C" factories and version getters are looked
// up; every method called on what they return is an inline forwarder in the
// headers to a virtual inside the library, so nothing links against
// libnvinfer. CUDA is the driver API alone, each entry point asked of
// cuGetProcAddress at the version its signature here matches: a shim built on
// CUDA 12 headers then gets the same functions from a CUDA 13 driver.
//
// One source for both builds: TensorRT 10 (Jetson, the 10.3 headers) and 11
// (PCs); the only difference the server sees is the network's typing and
// whether there is an FP16 flag.
#include "jl_trt.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda.h>
#include <dlfcn.h>

#include <atomic>
#include <cerrno>
#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <mutex>

#define JL_STR2(x) #x
#define JL_STR(x) JL_STR2(x)

namespace {

// --- errors ------------------------------------------------------------------

int say(char *err, size_t errlen, int code, const char *fmt, ...) {
  if (err != nullptr && errlen > 0) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(err, errlen, fmt, args);
    va_end(args);
  }
  return code;
}

// What TensorRT last logged as an error on this thread: the only reason it
// gives when a call returns NULL or false. Cleared before each such call.
thread_local char last_error[1024];

int trt_fail(char *err, size_t errlen, const char *call, const char *fallback) {
  return say(err, errlen, JL_TRT_ERROR, "%s: %s", call, last_error[0] ? last_error : fallback);
}

// Errors that leave the context unusable: every later call in it fails the
// same way, and only a new process gets a working one back (D15). Unknown is
// here too, since nothing says the context survived it.
bool is_sticky(CUresult r) {
  switch (r) {
  case CUDA_ERROR_ILLEGAL_ADDRESS:
  case CUDA_ERROR_LAUNCH_TIMEOUT:
  case CUDA_ERROR_HARDWARE_STACK_ERROR:
  case CUDA_ERROR_ILLEGAL_INSTRUCTION:
  case CUDA_ERROR_MISALIGNED_ADDRESS:
  case CUDA_ERROR_INVALID_ADDRESS_SPACE:
  case CUDA_ERROR_INVALID_PC:
  case CUDA_ERROR_LAUNCH_FAILED:
  case CUDA_ERROR_ASSERT:
  case CUDA_ERROR_ECC_UNCORRECTABLE:
  case CUDA_ERROR_NVLINK_UNCORRECTABLE:
  case CUDA_ERROR_EXTERNAL_DEVICE:
  case CUDA_ERROR_CONTEXT_IS_DESTROYED:
  case CUDA_ERROR_DEVICE_UNAVAILABLE:
  case CUDA_ERROR_DEINITIALIZED:
  case CUDA_ERROR_UNKNOWN:
    return true;
  default:
    return false;
  }
}

// --- the driver API --------------------------------------------------------------

typedef CUresult (*GetProcAddressV2)(const char *, void **, int, cuuint64_t, CUdriverProcAddressQueryResult *);
typedef CUresult (*GetProcAddressV1)(const char *, void **, int, cuuint64_t);

struct Cuda {
  CUresult (*Init)(unsigned);
  CUresult (*DriverGetVersion)(int *);
  CUresult (*DeviceGet)(CUdevice *, int);
  CUresult (*DeviceGetName)(char *, int, CUdevice);
  CUresult (*DeviceGetAttribute)(int *, CUdevice_attribute, CUdevice);
  CUresult (*DevicePrimaryCtxRetain)(CUcontext *, CUdevice);
  CUresult (*DevicePrimaryCtxRelease)(CUdevice);
  CUresult (*CtxSetCurrent)(CUcontext);
  CUresult (*MemGetInfo)(size_t *, size_t *);
  CUresult (*MemAlloc)(CUdeviceptr *, size_t);
  CUresult (*MemFree)(CUdeviceptr);
  CUresult (*MemHostAlloc)(void **, size_t, unsigned);
  CUresult (*MemFreeHost)(void *);
  CUresult (*MemcpyHtoDAsync)(CUdeviceptr, const void *, size_t, CUstream);
  CUresult (*MemcpyDtoHAsync)(void *, CUdeviceptr, size_t, CUstream);
  CUresult (*MemcpyDtoDAsync)(CUdeviceptr, CUdeviceptr, size_t, CUstream);
  CUresult (*MemsetD8Async)(CUdeviceptr, unsigned char, size_t, CUstream);
  CUresult (*StreamCreate)(CUstream *, unsigned);
  CUresult (*StreamSynchronize)(CUstream);
  CUresult (*StreamQuery)(CUstream);
  CUresult (*StreamIsCapturing)(CUstream, CUstreamCaptureStatus *);
  CUresult (*StreamDestroy)(CUstream);
  CUresult (*EventCreate)(CUevent *, unsigned);
  CUresult (*EventRecordWithFlags)(CUevent, CUstream, unsigned);
  CUresult (*EventSynchronize)(CUevent);
  CUresult (*EventQuery)(CUevent);
  CUresult (*EventElapsedTime)(float *, CUevent, CUevent);
  CUresult (*EventDestroy)(CUevent);
  CUresult (*StreamBeginCapture)(CUstream, CUstreamCaptureMode);
  CUresult (*StreamEndCapture)(CUstream, CUgraph *);
  CUresult (*GraphInstantiateWithFlags)(CUgraphExec *, CUgraph, unsigned long long);
  CUresult (*GraphLaunch)(CUgraphExec, CUstream);
  CUresult (*GraphDestroy)(CUgraph);
  CUresult (*GraphExecDestroy)(CUgraphExec);
  CUresult (*GetErrorName)(CUresult, const char **);
  CUresult (*GetErrorString)(CUresult, const char **);
};

// Each entry point, and the CUDA version whose signature the member above
// declares (cudaTypedefs.h's PFN_<name>_v<version>).
struct Entry {
  const char *name;
  int version;
  size_t offset;
};

#define JL_ENTRY(member, version) {"cu" #member, version, offsetof(Cuda, member)}
const Entry entries[] = {
    JL_ENTRY(Init, 2000),
    JL_ENTRY(DriverGetVersion, 2020),
    JL_ENTRY(DeviceGet, 2000),
    JL_ENTRY(DeviceGetName, 2000),
    JL_ENTRY(DeviceGetAttribute, 2000),
    JL_ENTRY(DevicePrimaryCtxRetain, 7000),
    JL_ENTRY(DevicePrimaryCtxRelease, 11000),
    JL_ENTRY(CtxSetCurrent, 4000),
    JL_ENTRY(MemGetInfo, 3020),
    JL_ENTRY(MemAlloc, 3020),
    JL_ENTRY(MemFree, 3020),
    JL_ENTRY(MemHostAlloc, 2020),
    JL_ENTRY(MemFreeHost, 2000),
    JL_ENTRY(MemcpyHtoDAsync, 3020),
    JL_ENTRY(MemcpyDtoHAsync, 3020),
    JL_ENTRY(MemcpyDtoDAsync, 3020),
    JL_ENTRY(MemsetD8Async, 3020),
    JL_ENTRY(StreamCreate, 2000),
    JL_ENTRY(StreamSynchronize, 2000),
    JL_ENTRY(StreamQuery, 2000),
    JL_ENTRY(StreamIsCapturing, 10000),
    JL_ENTRY(StreamDestroy, 4000),
    JL_ENTRY(EventCreate, 2000),
    JL_ENTRY(EventRecordWithFlags, 11010),
    JL_ENTRY(EventSynchronize, 2000),
    JL_ENTRY(EventQuery, 2000),
    JL_ENTRY(EventElapsedTime, 2000),
    JL_ENTRY(EventDestroy, 4000),
    JL_ENTRY(StreamBeginCapture, 10010),
    JL_ENTRY(StreamEndCapture, 10000),
    JL_ENTRY(GraphInstantiateWithFlags, 11040),
    JL_ENTRY(GraphLaunch, 10000),
    JL_ENTRY(GraphDestroy, 10000),
    JL_ENTRY(GraphExecDestroy, 10000),
    JL_ENTRY(GetErrorName, 6000),
    JL_ENTRY(GetErrorString, 6000),
};
#undef JL_ENTRY

// --- TensorRT's callbacks ------------------------------------------------------------

// TensorRT keeps the first logger it is given for the whole process, so there
// is one, and jl_trt_set_logger changes where it forwards.
class Logger final : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char *msg) noexcept override {
    int level = static_cast<int>(severity);
    if (level <= JL_TRT_LOG_ERROR) {
      snprintf(last_error, sizeof last_error, "%s", msg);
    }
    std::lock_guard<std::mutex> hold(lock);
    if (fn != nullptr && level <= min) {
      fn(ctx, level, msg);
    }
  }

  std::mutex lock;
  jl_trt_log_fn fn = nullptr;
  void *ctx = nullptr;
  int min = JL_TRT_LOG_WARNING;
};

Logger logger;

// What TensorRT's factories expect: the ILogger itself, as their inline
// wrappers pass it.
void *trt_logger() {
  return static_cast<nvinfer1::ILogger *>(&logger);
}

class Monitor final : public nvinfer1::IProgressMonitor {
public:
  void phaseStart(const char *phase, const char *parent, int32_t steps) noexcept override {
    emit(JL_TRT_PHASE_START, phase, parent, steps);
  }

  bool stepComplete(const char *phase, int32_t step) noexcept override {
    emit(JL_TRT_PHASE_STEP, phase, nullptr, step);
    return true;
  }

  void phaseFinish(const char *phase) noexcept override {
    emit(JL_TRT_PHASE_FINISH, phase, nullptr, 0);
  }

  std::mutex lock;
  jl_trt_progress_fn fn = nullptr;
  void *ctx = nullptr;

private:
  // TensorRT may report from several threads; Swift hears one at a time
  void emit(int event, const char *phase, const char *parent, int value) {
    std::lock_guard<std::mutex> hold(lock);
    if (fn != nullptr) {
      fn(ctx, event, phase, parent, value);
    }
  }
};

// The context each thread last made current, and for which open: a context
// released and retained again can come back at the same address.
struct Current {
  CUcontext ctx;
  uint64_t generation;
};
thread_local Current current = {nullptr, 0};
std::atomic<uint64_t> generations{0};

int onnx_type(nvinfer1::DataType type) {
  switch (type) {
  case nvinfer1::DataType::kFLOAT:
    return JL_TRT_FLOAT;
  case nvinfer1::DataType::kHALF:
    return JL_TRT_FLOAT16;
  case nvinfer1::DataType::kINT8:
    return JL_TRT_INT8;
  case nvinfer1::DataType::kINT32:
    return JL_TRT_INT32;
  case nvinfer1::DataType::kINT64:
    return JL_TRT_INT64;
  case nvinfer1::DataType::kBOOL:
    return JL_TRT_BOOL;
  case nvinfer1::DataType::kUINT8:
    return JL_TRT_UINT8;
  default:
    return 0;
  }
}

// onnx-tensorrt's errorCodeStr, which the headers do not carry
const char *parser_code(nvonnxparser::ErrorCode code) {
  static const char *const names[] = {
      "SUCCESS",           "INTERNAL_ERROR",        "MEM_ALLOC_FAILED",          "MODEL_DESERIALIZE_FAILED",
      "INVALID_VALUE",     "INVALID_GRAPH",         "INVALID_NODE",              "UNSUPPORTED_GRAPH",
      "UNSUPPORTED_NODE",  "UNSUPPORTED_NODE_ATTR", "UNSUPPORTED_NODE_INPUT",    "UNSUPPORTED_NODE_DATATYPE",
      "UNSUPPORTED_NODE_DYNAMIC", "UNSUPPORTED_NODE_SHAPE", "REFIT_FAILED",
  };
  size_t i = static_cast<size_t>(code);
  return i < sizeof names / sizeof names[0] ? names[i] : "UNKNOWN";
}

const char *or_empty(const char *s) {
  return s != nullptr ? s : "";
}

bool write_file(const char *path, const void *data, size_t size) {
  FILE *f = fopen(path, "wb");
  if (f == nullptr) {
    return false;
  }
  bool ok = fwrite(data, 1, size, f) == size;
  return fclose(f) == 0 && ok;
}

} // namespace

struct jl_trt {
  void *cuda_lib = nullptr;
  Cuda cu = {};
  CUdevice device = 0;
  CUcontext ctx = nullptr;
  uint64_t generation = 0;

  void *(*create_runtime)(void *, int32_t) = nullptr;
  void *(*create_builder)(void *, int32_t) = nullptr;
  void *(*create_parser)(void *, void *, int) = nullptr;

  std::atomic<int> sticky{0};
  std::mutex sticky_lock;
  char sticky_message[512] = {0};

  char device_name[256] = {0};
  jl_trt_info info = {};
};

struct jl_trt_engine {
  jl_trt *trt;
  nvinfer1::IRuntime *runtime;
  nvinfer1::ICudaEngine *engine;
};

struct jl_trt_context {
  jl_trt *trt;
  nvinfer1::IExecutionContext *context;
};

struct jl_trt_build {
  jl_trt *trt;
  nvinfer1::IBuilder *builder = nullptr;
  nvinfer1::INetworkDefinition *network = nullptr;
  nvonnxparser::IParser *parser = nullptr;
  nvinfer1::IBuilderConfig *config = nullptr;
  nvinfer1::ITimingCache *cache = nullptr;
  Monitor monitor;
};

namespace {

CUstream cu(jl_trt_stream *s) {
  return reinterpret_cast<CUstream>(s);
}

CUevent cu(jl_trt_event *e) {
  return reinterpret_cast<CUevent>(e);
}

int cuda_fail(jl_trt *t, const char *call, CUresult r, char *err, size_t errlen) {
  const char *name = nullptr;
  const char *text = nullptr;
  if (t->cu.GetErrorName != nullptr) {
    t->cu.GetErrorName(r, &name);
    t->cu.GetErrorString(r, &text);
  }
  char message[512];
  snprintf(message, sizeof message, "%s: %s: %s", call, name ? name : "CUDA_ERROR_UNKNOWN", text ? text : "unknown error");
  if (!is_sticky(r)) {
    return say(err, errlen, JL_TRT_CUDA_ERROR, "%s", message);
  }
  {
    std::lock_guard<std::mutex> hold(t->sticky_lock);
    if (t->sticky.load() == 0) {
      snprintf(t->sticky_message, sizeof t->sticky_message, "%s", message);
      t->sticky.store(1);
    }
  }
  return say(err, errlen, JL_TRT_CUDA_STICKY, "%s", message);
}

// Every call that reaches CUDA or TensorRT starts here: fail at once once the
// context is broken, else make it current on this thread if it is not yet.
int enter(jl_trt *t, char *err, size_t errlen) {
  if (t->sticky.load(std::memory_order_acquire) != 0) {
    std::lock_guard<std::mutex> hold(t->sticky_lock);
    return say(err, errlen, JL_TRT_CUDA_STICKY, "%s (latched)", t->sticky_message);
  }
  if (current.ctx != t->ctx || current.generation != t->generation) {
    CUresult r = t->cu.CtxSetCurrent(t->ctx);
    if (r != CUDA_SUCCESS) {
      return cuda_fail(t, "cuCtxSetCurrent", r, err, errlen);
    }
    current = {t->ctx, t->generation};
  }
  last_error[0] = '\0';
  return JL_TRT_OK;
}

// For the calls that return nothing: current if it can be, never failing.
void enter_quietly(jl_trt *t) {
  if (t->sticky.load() == 0 && (current.ctx != t->ctx || current.generation != t->generation) &&
      t->cu.CtxSetCurrent(t->ctx) == CUDA_SUCCESS) {
    current = {t->ctx, t->generation};
  }
}

#define JL_CU(call, ...)                                    \
  do {                                                      \
    CUresult r_ = t->cu.call(__VA_ARGS__);                  \
    if (r_ != CUDA_SUCCESS) {                               \
      return cuda_fail(t, "cu" #call, r_, err, errlen);     \
    }                                                       \
  } while (0)

#define JL_ENTER(t)                        \
  do {                                     \
    int rc_ = enter((t), err, errlen);     \
    if (rc_ != JL_TRT_OK) {                \
      return rc_;                          \
    }                                      \
  } while (0)

int open_cuda(jl_trt *t, int device, char *err, size_t errlen) {
  t->cuda_lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  if (t->cuda_lib == nullptr) {
    return say(err, errlen, JL_TRT_UNAVAILABLE, "no CUDA driver: %s", dlerror());
  }
  // the _v2 form exists from CUDA 12.0; the first from 11.3
  auto v2 = reinterpret_cast<GetProcAddressV2>(dlsym(t->cuda_lib, "cuGetProcAddress_v2"));
  auto v1 = reinterpret_cast<GetProcAddressV1>(dlsym(t->cuda_lib, "cuGetProcAddress"));
  if (v2 == nullptr && v1 == nullptr) {
    return say(err, errlen, JL_TRT_UNAVAILABLE, "this CUDA driver is older than 11.3: it has no cuGetProcAddress");
  }
  for (const Entry &e : entries) {
    void *fn = nullptr;
    CUresult r = v2 != nullptr ? v2(e.name, &fn, e.version, CU_GET_PROC_ADDRESS_LEGACY_STREAM, nullptr)
                               : v1(e.name, &fn, e.version, CU_GET_PROC_ADDRESS_LEGACY_STREAM);
    if (r != CUDA_SUCCESS || fn == nullptr) {
      return say(err, errlen, JL_TRT_UNAVAILABLE, "this CUDA driver has no %s (version %d)", e.name, e.version);
    }
    memcpy(reinterpret_cast<char *>(&t->cu) + e.offset, &fn, sizeof fn);
  }
  CUresult r = t->cu.Init(0);
  if (r == CUDA_SUCCESS) {
    r = t->cu.DeviceGet(&t->device, device);
  }
  if (r != CUDA_SUCCESS) {
    cuda_fail(t, r == CUDA_ERROR_INVALID_DEVICE ? "cuDeviceGet" : "cuInit", r, err, errlen);
    return JL_TRT_UNAVAILABLE;
  }
  jl_trt_info &info = t->info;
  info.device = device;
  JL_CU(DriverGetVersion, &info.cuda_driver);
  JL_CU(DeviceGetName, t->device_name, static_cast<int>(sizeof t->device_name), t->device);
  JL_CU(DeviceGetAttribute, &info.cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, t->device);
  JL_CU(DeviceGetAttribute, &info.cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, t->device);
  JL_CU(DevicePrimaryCtxRetain, &t->ctx, t->device);
  t->generation = ++generations;
  return enter(t, err, errlen);
}

template <typename T> bool resolve(void *lib, const char *name, T *out) {
  *out = reinterpret_cast<T>(dlsym(lib, name));
  return *out != nullptr;
}

int open_tensorrt(jl_trt *t, char *err, size_t errlen) {
  const char *name = "libnvinfer.so." JL_STR(NV_TENSORRT_MAJOR);
  void *lib = dlopen(name, RTLD_NOW | RTLD_LOCAL);
  if (lib == nullptr) {
    return say(err, errlen, JL_TRT_UNAVAILABLE, "no TensorRT %d: %s", NV_TENSORRT_MAJOR, dlerror());
  }
  int32_t (*major)() = nullptr, (*minor)() = nullptr, (*patch)() = nullptr, (*build)() = nullptr;
  if (!resolve(lib, "createInferRuntime_INTERNAL", &t->create_runtime) ||
      !resolve(lib, "createInferBuilder_INTERNAL", &t->create_builder) ||
      !resolve(lib, "getInferLibMajorVersion", &major) || !resolve(lib, "getInferLibMinorVersion", &minor) ||
      !resolve(lib, "getInferLibPatchVersion", &patch) || !resolve(lib, "getInferLibBuildVersion", &build)) {
    return say(err, errlen, JL_TRT_UNAVAILABLE, "%s lacks an entry point jetlink needs: %s", name, dlerror());
  }
  jl_trt_info &info = t->info;
  info.major = major();
  info.minor = minor();
  info.patch = patch();
  info.build = build();
  // A newer minor runs what older headers compiled; an older one may lack
  // what they call. The build number is a respin of the same release.
  long have = NV_TENSORRT_VERSION_INT(info.major, info.minor, info.patch);
  if (info.major != NV_TENSORRT_MAJOR || have < static_cast<long>(NV_TENSORRT_VERSION)) {
    return say(err, errlen, JL_TRT_UNAVAILABLE, "TensorRT %d.%d.%d.%d is loaded; this build needs %d.x from %d.%d.%d on",
               info.major, info.minor, info.patch, info.build, NV_TENSORRT_MAJOR, NV_TENSORRT_MAJOR,
               NV_TENSORRT_MINOR, NV_TENSORRT_PATCH);
  }
  // Needed only to build, so a missing parser is jl_trt_build_create's error.
  void *parser = dlopen("libnvonnxparser.so." JL_STR(NV_TENSORRT_MAJOR), RTLD_NOW | RTLD_LOCAL);
  if (parser != nullptr) {
    resolve(parser, "createNvOnnxParser_INTERNAL", &t->create_parser);
  }
  // Python registered the plugins whenever the library had them, so a plan
  // with a plugin layer must still load here. Absent is fine: jetlink's
  // models use none.
  void *plugins = dlopen("libnvinfer_plugin.so." JL_STR(NV_TENSORRT_MAJOR), RTLD_NOW | RTLD_LOCAL);
  bool (*init_plugins)(void *, const char *) = nullptr;
  if (plugins != nullptr && resolve(plugins, "initLibNvInferPlugins", &init_plugins)) {
    info.plugins = init_plugins(trt_logger(), "") ? 1 : 0;
  }
  return JL_TRT_OK;
}

} // namespace

// --- library ---------------------------------------------------------------------

extern "C" {

int jl_trt_open(int device, jl_trt **out, char *err, size_t errlen) {
  *out = nullptr;
  jl_trt *t = new (std::nothrow) jl_trt();
  if (t == nullptr) {
    return say(err, errlen, JL_TRT_ERROR, "out of memory");
  }
  jl_trt_get_info(nullptr, &t->info);
  int rc = open_cuda(t, device, err, errlen);
  if (rc == JL_TRT_OK) {
    rc = open_tensorrt(t, err, errlen);
  }
  if (rc != JL_TRT_OK) {
    jl_trt_close(t);
    return rc;
  }
  t->info.device_name = t->device_name;
  *out = t;
  return JL_TRT_OK;
}

void jl_trt_close(jl_trt *t) {
  if (t == nullptr) {
    return;
  }
  if (t->ctx != nullptr) {
    t->cu.DevicePrimaryCtxRelease(t->device);
  }
  if (current.ctx == t->ctx) {
    current = {nullptr, 0};
  }
  // the libraries stay loaded: TensorRT's process-wide logger points here
  delete t;
}

void jl_trt_get_info(const jl_trt *t, jl_trt_info *out) {
  if (t != nullptr) {
    *out = t->info;
    return;
  }
  *out = jl_trt_info{};
  out->header_major = NV_TENSORRT_MAJOR;
  out->header_minor = NV_TENSORRT_MINOR;
  out->header_patch = NV_TENSORRT_PATCH;
  out->header_build = NV_TENSORRT_BUILD;
  out->strongly_typed = NV_TENSORRT_MAJOR >= 11 ? 1 : 0;
  out->device_name = "";
}

int jl_trt_mem_info(jl_trt *t, size_t *free_bytes, size_t *total_bytes, char *err, size_t errlen) {
  JL_ENTER(t);
  JL_CU(MemGetInfo, free_bytes, total_bytes);
  return JL_TRT_OK;
}

void jl_trt_set_logger(jl_trt *t, int min_severity, jl_trt_log_fn fn, void *ctx) {
  (void)t;
  std::lock_guard<std::mutex> hold(logger.lock);
  logger.fn = fn;
  logger.ctx = ctx;
  logger.min = min_severity;
}

int jl_trt_sticky(const jl_trt *t) {
  return t->sticky.load(std::memory_order_acquire);
}

// --- memory ------------------------------------------------------------------------

int jl_trt_mem_alloc(jl_trt *t, size_t size, jl_trt_dptr *out, char *err, size_t errlen) {
  *out = 0;
  JL_ENTER(t);
  CUdeviceptr p = 0;
  JL_CU(MemAlloc, &p, size);
  *out = static_cast<jl_trt_dptr>(p);
  return JL_TRT_OK;
}

void jl_trt_mem_free(jl_trt *t, jl_trt_dptr ptr) {
  if (ptr != 0) {
    enter_quietly(t);
    t->cu.MemFree(static_cast<CUdeviceptr>(ptr));
  }
}

int jl_trt_host_alloc(jl_trt *t, size_t size, void **out, char *err, size_t errlen) {
  *out = nullptr;
  JL_ENTER(t);
  // cudaHostAllocDefault, as Python allocated them
  JL_CU(MemHostAlloc, out, size, 0);
  return JL_TRT_OK;
}

void jl_trt_host_free(jl_trt *t, void *ptr) {
  if (ptr != nullptr) {
    enter_quietly(t);
    t->cu.MemFreeHost(ptr);
  }
}

int jl_trt_copy_h2d(jl_trt *t, jl_trt_dptr dst, const void *src, size_t size, jl_trt_stream *stream, char *err,
                    size_t errlen) {
  JL_ENTER(t);
  JL_CU(MemcpyHtoDAsync, static_cast<CUdeviceptr>(dst), src, size, cu(stream));
  return JL_TRT_OK;
}

int jl_trt_copy_d2h(jl_trt *t, void *dst, jl_trt_dptr src, size_t size, jl_trt_stream *stream, char *err,
                    size_t errlen) {
  JL_ENTER(t);
  JL_CU(MemcpyDtoHAsync, dst, static_cast<CUdeviceptr>(src), size, cu(stream));
  return JL_TRT_OK;
}

int jl_trt_copy_d2d(jl_trt *t, jl_trt_dptr dst, jl_trt_dptr src, size_t size, jl_trt_stream *stream, char *err,
                    size_t errlen) {
  JL_ENTER(t);
  JL_CU(MemcpyDtoDAsync, static_cast<CUdeviceptr>(dst), static_cast<CUdeviceptr>(src), size, cu(stream));
  return JL_TRT_OK;
}

int jl_trt_memset(jl_trt *t, jl_trt_dptr dst, uint8_t value, size_t size, jl_trt_stream *stream, char *err,
                  size_t errlen) {
  JL_ENTER(t);
  JL_CU(MemsetD8Async, static_cast<CUdeviceptr>(dst), value, size, cu(stream));
  return JL_TRT_OK;
}

// --- streams and events ----------------------------------------------------------------

int jl_trt_stream_create(jl_trt *t, jl_trt_stream **out, char *err, size_t errlen) {
  *out = nullptr;
  JL_ENTER(t);
  CUstream s = nullptr;
  // CU_STREAM_DEFAULT, as cudaStreamCreate
  JL_CU(StreamCreate, &s, 0);
  *out = reinterpret_cast<jl_trt_stream *>(s);
  return JL_TRT_OK;
}

int jl_trt_stream_sync(jl_trt *t, jl_trt_stream *stream, char *err, size_t errlen) {
  JL_ENTER(t);
  JL_CU(StreamSynchronize, cu(stream));
  return JL_TRT_OK;
}

void jl_trt_stream_destroy(jl_trt *t, jl_trt_stream *stream) {
  if (stream != nullptr) {
    enter_quietly(t);
    t->cu.StreamDestroy(cu(stream));
  }
}

int jl_trt_event_create(jl_trt *t, unsigned flags, jl_trt_event **out, char *err, size_t errlen) {
  *out = nullptr;
  JL_ENTER(t);
  CUevent e = nullptr;
  JL_CU(EventCreate, &e, flags);
  *out = reinterpret_cast<jl_trt_event *>(e);
  return JL_TRT_OK;
}

int jl_trt_event_record(jl_trt *t, jl_trt_event *event, jl_trt_stream *stream, unsigned flags, char *err,
                        size_t errlen) {
  JL_ENTER(t);
  JL_CU(EventRecordWithFlags, cu(event), cu(stream), flags);
  return JL_TRT_OK;
}

int jl_trt_event_sync(jl_trt *t, jl_trt_event *event, char *err, size_t errlen) {
  JL_ENTER(t);
  JL_CU(EventSynchronize, cu(event));
  return JL_TRT_OK;
}

int jl_trt_event_query(jl_trt *t, jl_trt_event *event, char *err, size_t errlen) {
  JL_ENTER(t);
  CUresult r = t->cu.EventQuery(cu(event));
  if (r == CUDA_ERROR_NOT_READY) {
    return JL_TRT_NOT_READY;
  }
  if (r != CUDA_SUCCESS) {
    return cuda_fail(t, "cuEventQuery", r, err, errlen);
  }
  return JL_TRT_OK;
}

int jl_trt_event_elapsed(jl_trt *t, jl_trt_event *start, jl_trt_event *end, float *ms, char *err, size_t errlen) {
  JL_ENTER(t);
  JL_CU(EventElapsedTime, ms, cu(start), cu(end));
  return JL_TRT_OK;
}

void jl_trt_event_destroy(jl_trt *t, jl_trt_event *event) {
  if (event != nullptr) {
    enter_quietly(t);
    t->cu.EventDestroy(cu(event));
  }
}

// --- graphs ----------------------------------------------------------------------------

int jl_trt_capture_begin(jl_trt *t, jl_trt_stream *stream, char *err, size_t errlen) {
  JL_ENTER(t);
  JL_CU(StreamBeginCapture, cu(stream), CU_STREAM_CAPTURE_MODE_THREAD_LOCAL);
  return JL_TRT_OK;
}

int jl_trt_capture_end(jl_trt *t, jl_trt_stream *stream, jl_trt_graph **out, char *err, size_t errlen) {
  *out = nullptr;
  JL_ENTER(t);
  CUgraph g = nullptr;
  CUresult r = t->cu.StreamEndCapture(cu(stream), &g);
  if (r != CUDA_SUCCESS) {
    // an invalidated capture can still hand back a partial graph
    if (g != nullptr) {
      t->cu.GraphDestroy(g);
    }
    return cuda_fail(t, "cuStreamEndCapture", r, err, errlen);
  }
  *out = reinterpret_cast<jl_trt_graph *>(g);
  return JL_TRT_OK;
}

int jl_trt_graph_instantiate(jl_trt *t, jl_trt_graph *graph, jl_trt_graph_exec **out, char *err, size_t errlen) {
  *out = nullptr;
  JL_ENTER(t);
  CUgraphExec x = nullptr;
  JL_CU(GraphInstantiateWithFlags, &x, reinterpret_cast<CUgraph>(graph), 0);
  *out = reinterpret_cast<jl_trt_graph_exec *>(x);
  return JL_TRT_OK;
}

int jl_trt_graph_launch(jl_trt *t, jl_trt_graph_exec *exec, jl_trt_stream *stream, char *err, size_t errlen) {
  JL_ENTER(t);
  JL_CU(GraphLaunch, reinterpret_cast<CUgraphExec>(exec), cu(stream));
  return JL_TRT_OK;
}

void jl_trt_graph_destroy(jl_trt *t, jl_trt_graph *graph) {
  if (graph != nullptr) {
    enter_quietly(t);
    t->cu.GraphDestroy(reinterpret_cast<CUgraph>(graph));
  }
}

void jl_trt_graph_exec_destroy(jl_trt *t, jl_trt_graph_exec *exec) {
  if (exec != nullptr) {
    enter_quietly(t);
    t->cu.GraphExecDestroy(reinterpret_cast<CUgraphExec>(exec));
  }
}

// --- runtime -------------------------------------------------------------------------------

int jl_trt_engine_deserialize(jl_trt *t, const void *plan, size_t size, jl_trt_engine **out, char *err,
                              size_t errlen) {
  *out = nullptr;
  JL_ENTER(t);
  auto *runtime = static_cast<nvinfer1::IRuntime *>(t->create_runtime(trt_logger(), NV_TENSORRT_VERSION));
  if (runtime == nullptr) {
    return trt_fail(err, errlen, "createInferRuntime", "TensorRT returned no runtime");
  }
  nvinfer1::ICudaEngine *engine = runtime->deserializeCudaEngine(plan, size);
  if (engine == nullptr) {
    delete runtime;
    return trt_fail(err, errlen, "deserializeCudaEngine", "TensorRT could not deserialize the plan");
  }
  *out = new jl_trt_engine{t, runtime, engine};
  return JL_TRT_OK;
}

void jl_trt_engine_destroy(jl_trt_engine *e) {
  if (e == nullptr) {
    return;
  }
  enter_quietly(e->trt);
  delete e->engine;
  delete e->runtime;
  delete e;
}

int jl_trt_engine_io_count(const jl_trt_engine *e) {
  return e->engine->getNbIOTensors();
}

int jl_trt_engine_io(const jl_trt_engine *e, int index, const char **name, int *is_input, int *type, int64_t *dims,
                     int *rank, char *err, size_t errlen) {
  int count = e->engine->getNbIOTensors();
  if (index < 0 || index >= count) {
    return say(err, errlen, JL_TRT_ERROR, "no IO tensor %d; the engine has %d", index, count);
  }
  const char *n = e->engine->getIOTensorName(index);
  nvinfer1::Dims shape = e->engine->getTensorShape(n);
  if (shape.nbDims < 0 || shape.nbDims > JL_TRT_MAX_DIMS) {
    return say(err, errlen, JL_TRT_ERROR, "tensor %s has %d dims", n, shape.nbDims);
  }
  *name = n;
  *is_input = e->engine->getTensorIOMode(n) == nvinfer1::TensorIOMode::kINPUT;
  *type = onnx_type(e->engine->getTensorDataType(n));
  *rank = shape.nbDims;
  for (int i = 0; i < shape.nbDims; i++) {
    dims[i] = shape.d[i];
  }
  return JL_TRT_OK;
}

int jl_trt_context_create(jl_trt_engine *e, jl_trt_context **out, char *err, size_t errlen) {
  *out = nullptr;
  JL_ENTER(e->trt);
  nvinfer1::IExecutionContext *context = e->engine->createExecutionContext();
  if (context == nullptr) {
    return trt_fail(err, errlen, "createExecutionContext", "TensorRT returned no execution context");
  }
  *out = new jl_trt_context{e->trt, context};
  return JL_TRT_OK;
}

void jl_trt_context_destroy(jl_trt_context *c) {
  if (c == nullptr) {
    return;
  }
  enter_quietly(c->trt);
  delete c->context;
  delete c;
}

int jl_trt_context_set_address(jl_trt_context *c, const char *name, jl_trt_dptr address, char *err, size_t errlen) {
  last_error[0] = '\0';
  if (!c->context->setTensorAddress(name, reinterpret_cast<void *>(static_cast<uintptr_t>(address)))) {
    return trt_fail(err, errlen, "setTensorAddress", "TensorRT refused the address");
  }
  return JL_TRT_OK;
}

int jl_trt_context_enqueue(jl_trt_context *c, jl_trt_stream *stream, char *err, size_t errlen) {
  jl_trt *t = c->trt;
  JL_ENTER(t);
  if (c->context->enqueueV3(reinterpret_cast<cudaStream_t>(stream))) {
    return JL_TRT_OK;
  }
  // A refusal may hide a CUDA error. Querying a capturing stream would end
  // the capture, so only one that is not capturing is asked.
  CUstreamCaptureStatus status = CU_STREAM_CAPTURE_STATUS_NONE;
  if (t->cu.StreamIsCapturing(cu(stream), &status) == CUDA_SUCCESS && status == CU_STREAM_CAPTURE_STATUS_NONE) {
    CUresult r = t->cu.StreamQuery(cu(stream));
    if (r != CUDA_SUCCESS && r != CUDA_ERROR_NOT_READY) {
      return cuda_fail(t, "enqueueV3", r, err, errlen);
    }
  }
  return trt_fail(err, errlen, "enqueueV3", "TensorRT refused to enqueue");
}

// --- builder -----------------------------------------------------------------------------

int jl_trt_build_create(jl_trt *t, jl_trt_build **out, char *err, size_t errlen) {
  *out = nullptr;
  if (t->create_parser == nullptr) {
    return say(err, errlen, JL_TRT_UNAVAILABLE, "no ONNX parser: libnvonnxparser.so.%d did not load", NV_TENSORRT_MAJOR);
  }
  JL_ENTER(t);
  jl_trt_build *b = new jl_trt_build();
  b->trt = t;
  b->builder = static_cast<nvinfer1::IBuilder *>(t->create_builder(trt_logger(), NV_TENSORRT_VERSION));
  if (b->builder == nullptr) {
    jl_trt_build_destroy(b);
    return trt_fail(err, errlen, "createInferBuilder", "TensorRT returned no builder");
  }
  // 0 on TensorRT 10 is a weakly typed network, precision then picked by the
  // FP16 flag: the build the car was validated on. On 11 every network is
  // strongly typed and the flag that asked for it is deprecated and ignored.
  b->network = b->builder->createNetworkV2(0);
  if (b->network != nullptr) {
    b->parser = static_cast<nvonnxparser::IParser *>(t->create_parser(b->network, trt_logger(), NV_ONNX_PARSER_VERSION));
  }
  if (b->parser != nullptr) {
    b->config = b->builder->createBuilderConfig();
  }
  if (b->config == nullptr) {
    jl_trt_build_destroy(b);
    return trt_fail(err, errlen, "createBuilderConfig", "TensorRT returned no network, parser or config");
  }
  *out = b;
  return JL_TRT_OK;
}

void jl_trt_build_destroy(jl_trt_build *b) {
  if (b == nullptr) {
    return;
  }
  enter_quietly(b->trt);
  // the reverse of creation; the config refers to the cache, the network to
  // the parser's weights
  delete b->config;
  delete b->cache;
  delete b->parser;
  delete b->network;
  delete b->builder;
  delete b;
}

int jl_trt_build_parse(jl_trt_build *b, const char *onnx_path, char *err, size_t errlen) {
  JL_ENTER(b->trt);
  if (b->parser->parseFromFile(onnx_path, 0)) {
    return JL_TRT_OK;
  }
  // str(ParserError) in TensorRT's Python bindings, one per line
  size_t used = 0;
  int n = b->parser->getNbErrors();
  if (err != nullptr && errlen > 0) {
    err[0] = '\0';
  }
  for (int i = 0; i < n && err != nullptr && used + 1 < errlen; i++) {
    const nvonnxparser::IParserError *e = b->parser->getError(i);
    if (e == nullptr) {
      continue;
    }
    const char *sep = i > 0 ? "\n" : "";
    int w;
    if (e->code() == nvonnxparser::ErrorCode::kMODEL_DESERIALIZE_FAILED ||
        e->code() == nvonnxparser::ErrorCode::kREFIT_FAILED) {
      w = snprintf(err + used, errlen - used, "%s(%s): %s: %s", sep, or_empty(e->func()), parser_code(e->code()),
                   or_empty(e->desc()));
    } else {
      w = snprintf(err + used, errlen - used, "%sIn node %d with name: %s and operator: %s (%s): %s: %s", sep, e->node(),
                   or_empty(e->nodeName()), or_empty(e->nodeOperator()), or_empty(e->func()), parser_code(e->code()),
                   or_empty(e->desc()));
    }
    if (w < 0) {
      break;
    }
    used += static_cast<size_t>(w);
  }
  if (n == 0) {
    say(err, errlen, JL_TRT_ERROR, "%s", last_error[0] ? last_error : "the ONNX parser failed without an error");
  }
  return JL_TRT_ERROR;
}

int jl_trt_build_layers(const jl_trt_build *b) {
  return b->network->getNbLayers();
}

int jl_trt_build_set_fp16(jl_trt_build *b, char *err, size_t errlen) {
#if NV_TENSORRT_MAJOR >= 11
  (void)b;
  return say(err, errlen, JL_TRT_ERROR, "TensorRT %d has no FP16 flag; precision follows the ONNX", NV_TENSORRT_MAJOR);
#else
  (void)err;
  (void)errlen;
  b->config->setFlag(nvinfer1::BuilderFlag::kFP16);
  return JL_TRT_OK;
#endif
}

void jl_trt_build_set_optimization_level(jl_trt_build *b, int level) {
  b->config->setBuilderOptimizationLevel(level);
}

void jl_trt_build_set_workspace(jl_trt_build *b, size_t bytes) {
  b->config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, bytes);
}

void jl_trt_build_set_progress(jl_trt_build *b, jl_trt_progress_fn fn, void *ctx) {
  {
    std::lock_guard<std::mutex> hold(b->monitor.lock);
    b->monitor.fn = fn;
    b->monitor.ctx = ctx;
  }
  b->config->setProgressMonitor(fn != nullptr ? &b->monitor : nullptr);
}

int jl_trt_build_set_timing_cache(jl_trt_build *b, const void *data, size_t size, char *err, size_t errlen) {
  JL_ENTER(b->trt);
  nvinfer1::ITimingCache *cache = b->config->createTimingCache(size > 0 ? data : nullptr, size);
  if (cache == nullptr) {
    return trt_fail(err, errlen, "createTimingCache", "TensorRT could not read the timing cache");
  }
  if (!b->config->setTimingCache(*cache, false)) {
    delete cache;
    return trt_fail(err, errlen, "setTimingCache", "the timing cache does not match this TensorRT and GPU");
  }
  // the config holds the new one now
  delete b->cache;
  b->cache = cache;
  return JL_TRT_OK;
}

int jl_trt_build_write_plan(jl_trt_build *b, const char *path, char *err, size_t errlen) {
  JL_ENTER(b->trt);
  nvinfer1::IHostMemory *plan = b->builder->buildSerializedNetwork(*b->network, *b->config);
  if (plan == nullptr) {
    return trt_fail(err, errlen, "buildSerializedNetwork", "TensorRT returned no engine");
  }
  bool ok = write_file(path, plan->data(), plan->size());
  delete plan;
  if (!ok) {
    return say(err, errlen, JL_TRT_ERROR, "cannot write %s: %s", path, strerror(errno));
  }
  return JL_TRT_OK;
}

int jl_trt_build_write_timing_cache(jl_trt_build *b, const char *path, char *err, size_t errlen) {
  if (b->cache == nullptr) {
    return say(err, errlen, JL_TRT_ERROR, "no timing cache is attached");
  }
  JL_ENTER(b->trt);
  nvinfer1::IHostMemory *blob = b->cache->serialize();
  if (blob == nullptr) {
    return trt_fail(err, errlen, "ITimingCache::serialize", "TensorRT returned nothing");
  }
  bool ok = write_file(path, blob->data(), blob->size());
  delete blob;
  if (!ok) {
    return say(err, errlen, JL_TRT_ERROR, "cannot write %s: %s", path, strerror(errno));
  }
  return JL_TRT_OK;
}

} // extern "C"
