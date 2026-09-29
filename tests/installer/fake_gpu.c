// A stand-in for NVIDIA's driver and TensorRT, for the installer scenarios
// that run the real server from a release tarball: enough of libcuda.so.1 and
// libnvinfer.so.N for the server to open a "GPU", report it and wait for the
// comma. Anything more, a stream or a build, fails as unsupported. scenarios.sh
// builds it twice:
//
//   cc -shared -fPIC -DFAKE_CUDA -o libcuda.so.1 fake_gpu.c
//   cc -shared -fPIC -DFAKE_TRT -DTRT_MAJOR=10 -DTRT_MINOR=16 -DTRT_PATCH=2
//     -DTRT_BUILD=10 -o libnvinfer.so.10 fake_gpu.c
//
// Test code only: nothing here runs outside tests/installer.
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#ifdef FAKE_CUDA

typedef int CUresult;
enum { OK = 0, INVALID_DEVICE = 101, NOT_SUPPORTED = 801 };
// cuDeviceGetAttribute's CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR and _MINOR
enum { CC_MAJOR = 75, CC_MINOR = 76 };

static int context;

static CUresult init(unsigned flags) {
  (void)flags;
  return OK;
}

static CUresult driver_version(int *version) {
  *version = 13020;
  return OK;
}

static CUresult device_get(int *device, int ordinal) {
  if (ordinal != 0) {
    return INVALID_DEVICE;
  }
  *device = 0;
  return OK;
}

static CUresult device_name(char *name, int length, int device) {
  (void)device;
  snprintf(name, (size_t)length, "Fake GPU");
  return OK;
}

static CUresult device_attribute(int *value, int attribute, int device) {
  (void)device;
  *value = attribute == CC_MAJOR ? 8 : attribute == CC_MINOR ? 7 : 0;
  return OK;
}

static CUresult context_retain(void **ctx, int device) {
  (void)device;
  *ctx = &context;
  return OK;
}

static CUresult context_release(int device) {
  (void)device;
  return OK;
}

static CUresult context_set(void *ctx) {
  (void)ctx;
  return OK;
}

static CUresult memory(size_t *free_bytes, size_t *total_bytes) {
  *free_bytes = (size_t)6 << 30;
  *total_bytes = (size_t)8 << 30;
  return OK;
}

static CUresult error_text(CUresult result, const char **text) {
  (void)result;
  *text = "CUDA_ERROR_NOT_SUPPORTED (the installer tests' stand-in GPU)";
  return OK;
}

static CUresult unsupported(void) {
  return NOT_SUPPORTED;
}

static const struct {
  const char *name;
  void *fn;
} entries[] = {
    {"cuInit", (void *)init},
    {"cuDriverGetVersion", (void *)driver_version},
    {"cuDeviceGet", (void *)device_get},
    {"cuDeviceGetName", (void *)device_name},
    {"cuDeviceGetAttribute", (void *)device_attribute},
    {"cuDevicePrimaryCtxRetain", (void *)context_retain},
    {"cuDevicePrimaryCtxRelease", (void *)context_release},
    {"cuCtxSetCurrent", (void *)context_set},
    {"cuMemGetInfo", (void *)memory},
    {"cuGetErrorName", (void *)error_text},
    {"cuGetErrorString", (void *)error_text},
};

// Every entry point the server asks for resolves; those it needs only to run
// a model answer "not supported".
CUresult cuGetProcAddress_v2(const char *symbol, void **fn, int version, unsigned long long flags, int *status) {
  (void)version;
  (void)flags;
  if (status != NULL) {
    *status = 0;
  }
  *fn = (void *)unsupported;
  for (size_t i = 0; i < sizeof entries / sizeof entries[0]; i++) {
    if (strcmp(entries[i].name, symbol) == 0) {
      *fn = entries[i].fn;
    }
  }
  return OK;
}

#endif

#ifdef FAKE_TRT

int getInferLibMajorVersion(void) {
  return TRT_MAJOR;
}

int getInferLibMinorVersion(void) {
  return TRT_MINOR;
}

int getInferLibPatchVersion(void) {
  return TRT_PATCH;
}

int getInferLibBuildVersion(void) {
  return TRT_BUILD;
}

// Never called until a model is built or loaded, which the scenarios do not do.
void *createInferRuntime_INTERNAL(void *logger, int version) {
  (void)logger;
  (void)version;
  return NULL;
}

void *createInferBuilder_INTERNAL(void *logger, int version) {
  (void)logger;
  (void)version;
  return NULL;
}

#endif
