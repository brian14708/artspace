#include "cuda_stub.h"

#include <dlfcn.h>

static struct library {
  int force_managed_memory;

  // functions
  cudaStubError_t (*cudaMalloc)(void **, size_t);
  cudaStubError_t (*cudaFree)(void *);
  cudaStubError_t (*cudaGetDeviceCount)(int *);
  cudaStubError_t (*cudaGetDeviceProperties)(struct cudaStubDeviceProp *, int);
  cudaStubError_t (*cudaMallocManaged)(void **, size_t, unsigned int);
} g_library;

cudaStubError_t cudaMalloc(void **devPtr, size_t size) {
  if (g_library.force_managed_memory && g_library.cudaMallocManaged != NULL) {
    if (cudaStubSuccess == g_library.cudaMallocManaged(
                               devPtr, size, 0x01 /* cudaMemAttachGlobal */)) {
      return cudaStubSuccess;
    }
    // fallthrough
  }
  if (g_library.cudaMalloc != NULL) {
    return g_library.cudaMalloc(devPtr, size);
  }
  return cudaStubErrorInitializationError;
}

cudaStubError_t cudaFree(void *devPtr) {
  if (g_library.cudaFree != NULL) {
    return g_library.cudaFree(devPtr);
  }
  return cudaStubErrorInitializationError;
}

cudaStubError_t cudaStubGetDeviceCount(int *count) {
  if (g_library.cudaGetDeviceCount != NULL) {
    return g_library.cudaGetDeviceCount(count);
  }
  return cudaStubErrorInitializationError;
}

cudaStubError_t cudaStubGetDeviceProperties(struct cudaStubDeviceProp *prop,
                                            int device) {
  if (g_library.cudaGetDeviceProperties != NULL) {
    return g_library.cudaGetDeviceProperties(prop, device);
  }
  return cudaStubErrorInitializationError;
}

void cudaStubInit() {
  void *handle;
  handle = dlopen("libcudart.so", RTLD_LAZY | RTLD_LOCAL);
  if (!handle) {
    return;
  }

  g_library.cudaMalloc = dlsym(handle, "cudaMalloc");
  g_library.cudaFree = dlsym(handle, "cudaFree");
  g_library.cudaGetDeviceCount = dlsym(handle, "cudaGetDeviceCount");
  g_library.cudaGetDeviceProperties = dlsym(handle, "cudaGetDeviceProperties");
  g_library.cudaMallocManaged = dlsym(handle, "cudaMallocManaged");
  dlerror();

  if (g_library.cudaMallocManaged) {
    g_library.force_managed_memory = 1;
  }
}
