#ifndef _UTILS_COMMON_H_
#define _UTILS_COMMON_H_


#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda.h>


#define DRIVER_API_CALL(apiFuncCall)                                         \
  do {                                                                       \
    CUresult _status = apiFuncCall;                                          \
    if (_status != CUDA_SUCCESS) {                                           \
      fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n",   \
        __FILE__, __LINE__, #apiFuncCall, _status);                          \
      exit(-1);                                                              \
    }                                                                        \
  } while (0)


#define RUNTIME_API_CALL(apiFuncCall)                                        \
  do {                                                                       \
    cudaError_t _status = apiFuncCall;                                       \
    if (_status != cudaSuccess) {                                            \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
        __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status));      \
      exit(-1);                                                              \
    }                                                                        \
  } while (0)


#define GPU_TEST_FOR(x)                              \
  do {                                               \
    char *env = getenv("HPCTOOLKIT_GPU_TEST_REP");   \
    size_t rep = env == NULL ? 1 : atoi(env);        \
    for (size_t i = 0; i < rep; ++i) {               \
      x;                                             \
    }                                                \
  } while (0)


static inline void cu_init_device(int device_num, CUdevice &device, CUcontext &context) {
  static int const LEN = 32;
  char device_name[LEN];

  DRIVER_API_CALL(cuInit(0));
  DRIVER_API_CALL(cuDeviceGet(&device, device_num));
  DRIVER_API_CALL(cuDeviceGetName(device_name, LEN, device));

  DRIVER_API_CALL(cuCtxGetCurrent(&context));
  if (context == NULL) {
    DRIVER_API_CALL(cuCtxCreate(&context, 0, device));
    DRIVER_API_CALL(cuCtxSetCurrent(context));
  }
}


static inline void cuda_init_device(int device_num) {
  RUNTIME_API_CALL(cudaDeviceReset());
  RUNTIME_API_CALL(cudaSetDevice(device_num));
}


#endif
