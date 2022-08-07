#include "cuda_stub.h"

cudaStubError_t cudaStubGetDeviceCount(int *count) {
  (void)(count);
  return cudaStubErrorInitializationError;
}

cudaStubError_t cudaStubGetDeviceProperties(struct cudaStubDeviceProp *prop,
                                            int device) {
  (void)(prop);
  (void)(device);
  return cudaStubErrorInitializationError;
}

void cudaStubInit() {}
