// TensorRT, and the CUDA it runs on, as plain C calls for Swift. TensorRT has
// only a C++ API, and Swift can neither call it without turning on C++
// interop nor subclass the ILogger and IProgressMonitor TensorRT calls back
// into. The calls are the ones the Python server made (trt/engine.py,
// trt/build.py, cudart.py), one for one; what to do with them stays in Swift,
// so the fake (host memory, no GPU) exercises every Swift path.
//
// Nothing is linked: jl_trt_open dlopens libcuda.so.1, libnvinfer.so.<major>
// and libnvonnxparser.so.<major>, so a build needs headers only and a machine
// without TensorRT still starts and can say why. CUDA is the driver API, whose
// handles are the runtime's too: a stream here is the cudaStream_t enqueueV3
// takes.
//
// A call that can fail returns JL_TRT_OK (0) or one of the codes below, and
// writes why into err: errlen bytes at most, always NUL terminated, truncated
// to fit; err may be NULL. A CUDA failure reads "<call>: <CUDA_ERROR_NAME>:
// <description>". When TensorRT fails without giving a reason (a NULL engine,
// a false), the shim asks the context whether a sticky CUDA error is behind
// it, and returns that as JL_TRT_CUDA_STICKY if so, since a failed
// deserialize must not pass for a bad plan; else err holds the last error
// TensorRT logged on the calling thread.
// Frees and destroys return nothing: close paths cannot act on an error, and
// a sticky one has already surfaced from the call that caused it.
//
// CUDA contexts are the shim's business: jl_trt_open retains the device's
// primary context, and every call that reaches CUDA or TensorRT first makes it
// current on the calling thread unless a thread-local says that context
// already is. Swift may call from any thread, Dispatch's included, without
// knowing contexts exist.
//
// Nothing on the way to CUDA takes a lock. A handle is used by one thread at
// a time; different handles may be used on different threads at once (a
// build beside a loaded engine), as TensorRT allows. Destroy everything made
// from a jl_trt before jl_trt_close, and a context before its engine.
#ifndef JL_TRT_H
#define JL_TRT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
  JL_TRT_OK = 0,
  // TensorRT, the ONNX parser or a file said no.
  JL_TRT_ERROR = 1,
  // A CUDA call failed and the context still works: out of memory, a bad
  // argument, a capture that was invalidated.
  JL_TRT_CUDA_ERROR = 2,
  // A CUDA error that breaks the context for good (CUDA's sticky errors: an
  // illegal address, a failed launch, a hardware exception) or a device that
  // is gone. Only a new process recovers (D15). It latches, atomically, since
  // a build thread and the session thread share it: from then on every call
  // on the same jl_trt returns it without reaching CUDA, and jl_trt_sticky
  // says so.
  JL_TRT_CUDA_STICKY = 3,
  // jl_trt_open only: no CUDA driver or no such device, or no TensorRT of the
  // major this shim was compiled for, or one older than its headers
  // (major.minor.patch; the build number is not compared).
  JL_TRT_UNAVAILABLE = 4,
  // jl_trt_event_query only: the work before the event has not finished.
  JL_TRT_NOT_READY = 5,
};

// TensorRT's data types, as ONNX numbers them, so they read the same as
// jl_ort.h's and ElementType's. Any other TensorRT type reads 0: jetlink does
// not stage it.
enum {
  JL_TRT_FLOAT = 1,
  JL_TRT_UINT8 = 2,
  JL_TRT_INT8 = 3,
  JL_TRT_INT32 = 6,
  JL_TRT_INT64 = 7,
  JL_TRT_BOOL = 9,
  JL_TRT_FLOAT16 = 10,
};

// ILogger's severities: lower is worse.
enum {
  JL_TRT_LOG_INTERNAL_ERROR = 0,
  JL_TRT_LOG_ERROR = 1,
  JL_TRT_LOG_WARNING = 2,
  JL_TRT_LOG_INFO = 3,
  JL_TRT_LOG_VERBOSE = 4,
};

// Room for an IO tensor's dims (nvinfer1::Dims::MAX_DIMS).
#define JL_TRT_MAX_DIMS 8

// The libraries, the device and its retained primary context.
typedef struct jl_trt jl_trt;
// An IRuntime and the ICudaEngine it deserialized.
typedef struct jl_trt_engine jl_trt_engine;
// An IExecutionContext.
typedef struct jl_trt_context jl_trt_context;
// An IBuilder with its network, ONNX parser, config, progress monitor and
// timing cache: one handle, because TensorRT wants them destroyed together
// and in order.
typedef struct jl_trt_build jl_trt_build;
// CUstream, CUevent, CUgraph, CUgraphExec.
typedef struct jl_trt_stream jl_trt_stream;
typedef struct jl_trt_event jl_trt_event;
typedef struct jl_trt_graph jl_trt_graph;
typedef struct jl_trt_graph_exec jl_trt_graph_exec;

// A device address (CUdeviceptr): an integer, so Swift cannot mistake it for
// memory it may touch.
typedef uint64_t jl_trt_dptr;

// A TensorRT log line at the logger's severity or worse. message lasts for
// the call only.
typedef void (*jl_trt_log_fn)(void *ctx, int severity, const char *message);

// IProgressMonitor's calls, as they come: which phase started (value is its
// step count, parent NULL for a root phase), took a step (value is the step)
// or finished (value 0, parent NULL). Finding the root phase is Swift's, as
// trt/build.py's _Monitor did it, so the fake can test it. The build cannot
// be stopped from here.
enum {
  JL_TRT_PHASE_START = 0,
  JL_TRT_PHASE_STEP = 1,
  JL_TRT_PHASE_FINISH = 2,
};
typedef void (*jl_trt_progress_fn)(void *ctx, int event, const char *phase, const char *parent, int value);

// Both callbacks may run on TensorRT's own threads; the shim makes each one's
// calls one at a time.

typedef struct {
  // The libnvinfer loaded (getInferLib{Major,Minor,Patch,Build}Version).
  // Plans and timing caches are valid for this exact build only.
  int major, minor, patch, build;
  // The TensorRT headers the shim was compiled against (NV_TENSORRT_*).
  int header_major, header_minor, header_patch, header_build;
  // 1 on the TensorRT 11 build: networks are strongly typed and there is no
  // FP16 flag. 0 on TensorRT 10: weakly typed plus the flag, the build the car
  // was validated on.
  int strongly_typed;
  // cuDriverGetVersion: 12060 for CUDA 12.6.
  int cuda_driver;
  // 1 when libnvinfer_plugin was found and its plugins registered, as Python
  // did, so a plan with a plugin layer deserializes. Worth a log line.
  int plugins;
  // The device: its index, name ("Orin") and compute capability (8, 7), the
  // device part of the cache tag. device_name lasts until jl_trt_close.
  int device;
  const char *device_name;
  int cc_major, cc_minor;
} jl_trt_info;

// Opens CUDA and TensorRT and retains `device`'s primary context. Looked up
// by name are only TensorRT's factories and version getters; every other
// TensorRT call is an inline virtual in its headers. libnvonnxparser is
// needed by jl_trt_build_create alone, so a machine without it still loads
// plans; libnvinfer_plugin is registered when present and never required.
// The libraries stay loaded for the life of the process.
int jl_trt_open(int device, jl_trt **out, char *err, size_t errlen);
// Releases the primary context.
void jl_trt_close(jl_trt *trt);

// With trt NULL, only what the shim was compiled for: the header_ fields and
// strongly_typed; the rest reads 0, and device_name "".
void jl_trt_get_info(const jl_trt *trt, jl_trt_info *out);

// cuMemGetInfo, which changes as engines load: free and total device memory.
int jl_trt_mem_info(jl_trt *trt, size_t *free_bytes, size_t *total_bytes, char *err, size_t errlen);

// Where TensorRT's log lines go: those at min_severity or worse. TensorRT
// keeps one logger per process, so the shim hands it one forwarder and the
// logger set last receives every line. Set it right after jl_trt_open, before
// any engine or build; fn NULL drops them.
void jl_trt_set_logger(jl_trt *trt, int min_severity, jl_trt_log_fn fn, void *ctx);

// Nonzero once a call on trt returned JL_TRT_CUDA_STICKY.
int jl_trt_sticky(const jl_trt *trt);

// --- memory ---------------------------------------------------------------

// cuMemAlloc. Not zeroed.
int jl_trt_mem_alloc(jl_trt *trt, size_t size, jl_trt_dptr *out, char *err, size_t errlen);
void jl_trt_mem_free(jl_trt *trt, jl_trt_dptr ptr);

// cuMemHostAlloc with no flags (cudaHostAllocDefault): page-locked memory the
// staging writes into directly, and that the copies below move without a
// bounce buffer. Not zeroed.
int jl_trt_host_alloc(jl_trt *trt, size_t size, void **out, char *err, size_t errlen);
void jl_trt_host_free(jl_trt *trt, void *ptr);

// Stream-ordered copies and a byte memset (cuMemcpy{HtoD,DtoH,DtoD}Async,
// cuMemsetD8Async). Host memory comes from jl_trt_host_alloc: page-locked, so
// the copy is asynchronous and a graph can replay it.
int jl_trt_copy_h2d(jl_trt *trt, jl_trt_dptr dst, const void *src, size_t size, jl_trt_stream *stream,
                    char *err, size_t errlen);
int jl_trt_copy_d2h(jl_trt *trt, void *dst, jl_trt_dptr src, size_t size, jl_trt_stream *stream, char *err,
                    size_t errlen);
int jl_trt_copy_d2d(jl_trt *trt, jl_trt_dptr dst, jl_trt_dptr src, size_t size, jl_trt_stream *stream,
                    char *err, size_t errlen);
int jl_trt_memset(jl_trt *trt, jl_trt_dptr dst, uint8_t value, size_t size, jl_trt_stream *stream, char *err,
                  size_t errlen);

// --- streams and events -----------------------------------------------------

// cuStreamCreate with default flags, as cudaStreamCreate.
int jl_trt_stream_create(jl_trt *trt, jl_trt_stream **out, char *err, size_t errlen);
int jl_trt_stream_sync(jl_trt *trt, jl_trt_stream *stream, char *err, size_t errlen);
void jl_trt_stream_destroy(jl_trt *trt, jl_trt_stream *stream);

// Event flags, CUDA's values. The reply event is BLOCKING_SYNC |
// DISABLE_TIMING (0x3): a default event's synchronize spins, which cost 37%
// of a core on the Orin. Leave DISABLE_TIMING off for a pair that
// jl_trt_event_elapsed times (pure GPU time, for the bench).
enum {
  JL_TRT_EVENT_BLOCKING_SYNC = 0x1,
  JL_TRT_EVENT_DISABLE_TIMING = 0x2,
};
// Record flag: inside a capture, becomes an event node the host can wait on
// when the graph replays, rather than a dependency between streams.
enum {
  JL_TRT_RECORD_EXTERNAL = 0x1,
};
int jl_trt_event_create(jl_trt *trt, unsigned flags, jl_trt_event **out, char *err, size_t errlen);
int jl_trt_event_record(jl_trt *trt, jl_trt_event *event, jl_trt_stream *stream, unsigned flags, char *err,
                        size_t errlen);
int jl_trt_event_sync(jl_trt *trt, jl_trt_event *event, char *err, size_t errlen);
// cuEventQuery, without waiting: JL_TRT_OK once the work before the event is
// done, JL_TRT_NOT_READY before, or an error.
int jl_trt_event_query(jl_trt *trt, jl_trt_event *event, char *err, size_t errlen);
// Milliseconds between two recorded timing events (cuEventElapsedTime).
int jl_trt_event_elapsed(jl_trt *trt, jl_trt_event *start, jl_trt_event *end, float *ms, char *err,
                         size_t errlen);
void jl_trt_event_destroy(jl_trt *trt, jl_trt_event *event);

// --- graphs -----------------------------------------------------------------

// Captures what is queued on `stream` between begin and end into a graph, in
// thread-local mode (trtexec's: global mode would stop a build on another
// thread from allocating). Begin, the calls captured and end run on one
// thread, with nothing that synchronizes in between. Enqueue once before the
// first capture, as TensorRT requires. After a failure inside a capture, still
// call end: it ends the capture, returns the error and leaves the stream
// usable.
int jl_trt_capture_begin(jl_trt *trt, jl_trt_stream *stream, char *err, size_t errlen);
int jl_trt_capture_end(jl_trt *trt, jl_trt_stream *stream, jl_trt_graph **out, char *err, size_t errlen);
int jl_trt_graph_instantiate(jl_trt *trt, jl_trt_graph *graph, jl_trt_graph_exec **out, char *err,
                             size_t errlen);
// Replays the graph on stream. The addresses it captured are baked in: every
// buffer stays where it is, and allocated, for as long as it is launched.
int jl_trt_graph_launch(jl_trt *trt, jl_trt_graph_exec *exec, jl_trt_stream *stream, char *err, size_t errlen);
void jl_trt_graph_destroy(jl_trt *trt, jl_trt_graph *graph);
void jl_trt_graph_exec_destroy(jl_trt *trt, jl_trt_graph_exec *exec);

// --- runtime ----------------------------------------------------------------

// deserializeCudaEngine over the caller's bytes, which TensorRT does not keep:
// the caller maps the plan read-only and unmaps it when this returns, so a
// 1.7 GB plan is never read into the heap. A plan from another TensorRT build
// fails here with JL_TRT_ERROR (D21); a sticky CUDA error, with its own code.
int jl_trt_engine_deserialize(jl_trt *trt, const void *plan, size_t size, jl_trt_engine **out, char *err,
                              size_t errlen);
void jl_trt_engine_destroy(jl_trt_engine *engine);

int jl_trt_engine_io_count(const jl_trt_engine *engine);
// One IO tensor: its name (the engine's, valid until it is destroyed),
// whether it is an input, its type (JL_TRT_*, 0 for one jetlink does not
// stage) and its dims (dims has room for JL_TRT_MAX_DIMS; a dynamic one is
// -1, which the caller refuses).
int jl_trt_engine_io(const jl_trt_engine *engine, int index, const char **name, int *is_input, int *type,
                     int64_t *dims, int *rank, char *err, size_t errlen);

// createExecutionContext, its scratch memory allocated now (kSTATIC).
int jl_trt_context_create(jl_trt_engine *engine, jl_trt_context **out, char *err, size_t errlen);
void jl_trt_context_destroy(jl_trt_context *context);
int jl_trt_context_set_address(jl_trt_context *context, const char *name, jl_trt_dptr address, char *err,
                               size_t errlen);
// enqueueV3. A refusal asks the context for a sticky error as above, unless
// the stream is capturing: that would end the capture.
int jl_trt_context_enqueue(jl_trt_context *context, jl_trt_stream *stream, char *err, size_t errlen);

// --- builder ----------------------------------------------------------------

// An IBuilder, its network and an ONNX parser over it. The network is weakly
// typed on TensorRT 10, where the FP16 flag picks the precision, and strongly
// typed on 11, where precision follows the ONNX: jl_trt_info.strongly_typed
// says which. JL_TRT_UNAVAILABLE without libnvonnxparser.
int jl_trt_build_create(jl_trt *trt, jl_trt_build **out, char *err, size_t errlen);
void jl_trt_build_destroy(jl_trt_build *build);

// parseFromFile: a path rather than bytes, so weights stored beside the file
// are found. On failure err holds every parser error, one per line, as
// Python's str(ParserError) prints them: give it a few KB.
int jl_trt_build_parse(jl_trt_build *build, const char *onnx_path, char *err, size_t errlen);
// The parsed network's layer count.
int jl_trt_build_layers(const jl_trt_build *build);

// BuilderFlag::kFP16. TensorRT 10 only: JL_TRT_ERROR on 11, which has no
// such flag.
int jl_trt_build_set_fp16(jl_trt_build *build, char *err, size_t errlen);
void jl_trt_build_set_optimization_level(jl_trt_build *build, int level);
// The workspace pool's limit: a ceiling the tactics must fit under, not an
// allocation.
void jl_trt_build_set_workspace(jl_trt_build *build, size_t bytes);
void jl_trt_build_set_progress(jl_trt_build *build, jl_trt_progress_fn fn, void *ctx);

// A timing cache from an earlier build's bytes (NULL and 0 for an empty one),
// attached with mismatches not ignored. TensorRT copies the bytes. Nonzero
// when either step fails (another build's cache, a truncated one): then
// nothing is attached, and the caller attaches an empty cache, which the
// build fills and jl_trt_build_write_timing_cache saves. Python kept the
// unusable one, so its builds stayed cold.
int jl_trt_build_set_timing_cache(jl_trt_build *build, const void *data, size_t size, char *err,
                                  size_t errlen);

// buildSerializedNetwork, the plan written to path; the caller stages it and
// moves it into place. Blocks for the whole build (minutes for a big model)
// while progress arrives. A NULL plan is JL_TRT_ERROR.
int jl_trt_build_write_plan(jl_trt_build *build, const char *path, char *err, size_t errlen);
// The attached timing cache, which the build added to, serialized to path.
// The caller writes a .tmp and renames it, so a killed build never leaves a
// truncated cache behind.
int jl_trt_build_write_timing_cache(jl_trt_build *build, const char *path, char *err, size_t errlen);

#ifdef __cplusplus
}
#endif

#endif
