#include "cuda_stub.h"

#include <windows.h>

static struct library {
  cudaStubError_t(__cdecl *cudaGetDeviceCount)(int *);
  cudaStubError_t(__cdecl *cudaGetDeviceProperties)(struct cudaStubDeviceProp *,
                                                    int);
} g_library;

cudaStubError_t __cdecl cudaStubGetDeviceCount(int *count) {
  if (g_library.cudaGetDeviceCount != NULL) {
    return g_library.cudaGetDeviceCount(count);
  }
  return cudaStubErrorInitializationError;
}

cudaStubError_t __cdecl cudaStubGetDeviceProperties(
    struct cudaStubDeviceProp *prop, int device) {
  if (g_library.cudaGetDeviceProperties != NULL) {
    return g_library.cudaGetDeviceProperties(prop, device);
  }
  return cudaStubErrorInitializationError;
}

void __cdecl cudaStubInit() {
  HMODULE handle = LoadLibrary(L"cudart64_110.dll");
  if (!handle) {
    handle = LoadLibrary(L"cudart64_102.dll");
  }
  if (!handle) {
    handle = LoadLibrary(L"cudart64_101.dll");
  }
  if (!handle) {
    handle = LoadLibrary(L"cudart64_100.dll");
  }
  if (!handle) {
    return;
  }

  g_library.cudaGetDeviceCount = GetProcAddress(handle, "cudaGetDeviceCount");
  g_library.cudaGetDeviceProperties =
      GetProcAddress(handle, "cudaGetDeviceProperties");
}
