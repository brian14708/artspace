#include <stddef.h>

typedef enum cudaStubError {
  cudaStubSuccess = 0,
  cudaStubErrorInitializationError = 3,
  cudaStubErrorUnknown = 999,
} cudaStubError_t;

struct cudaStubDeviceProp {
  char name[256];
  struct CUuuid_st {
    char bytes[16];
  } uuid;
  char luid[8];
  unsigned int luidDeviceNodeMask;
  size_t totalGlobalMem;
};

cudaStubError_t cudaStubGetDeviceCount(int *count);

cudaStubError_t cudaStubGetDeviceProperties(struct cudaStubDeviceProp *prop,
                                            int device);

void cudaStubInit();