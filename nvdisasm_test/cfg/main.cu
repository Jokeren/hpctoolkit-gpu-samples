#include <cstdio>
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>


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


static size_t N = 1000;
static size_t iter1 = 200;
static size_t iter2 = 400;


void init(int *p, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    p[i] = i;
  }
}


void output(int *p, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    printf("index %zu: %d\n", i, p[i]);
  }
}


int main(int argc, char *argv[]) {
  // Init device
  CUcontext context;
  CUdevice device;

  int device_num = 0;
  if (argc != 1) {
    device_num = atoi(argv[1]);
  }

  DRIVER_API_CALL(cuInit(0));
  DRIVER_API_CALL(cuDeviceGet(&device, device_num));
  RUNTIME_API_CALL(cudaSetDevice(0));
  DRIVER_API_CALL(cuCtxCreate(&context, 0, device));

  #pragma omp parallel
  {
    int l[N], r[N], p[N];
    CUdeviceptr dl, dr, dp;

    init(l, N);
    init(r, N);

    size_t threads = 256;
    size_t blocks = (N - 1) / threads + 1;

    DRIVER_API_CALL(cuCtxSetCurrent(context));

    CUmodule moduleAdd;
    CUfunction vecAdd;
    DRIVER_API_CALL(cuModuleLoad(&moduleAdd, "vecAdd.cubin"));
    DRIVER_API_CALL(cuModuleGetFunction(&vecAdd, moduleAdd, "vecAdd"));

    DRIVER_API_CALL(cuMemAlloc(&dl, N * sizeof(int)));
    DRIVER_API_CALL(cuMemAlloc(&dr, N * sizeof(int)));
    DRIVER_API_CALL(cuMemAlloc(&dp, N * sizeof(int)));
    DRIVER_API_CALL(cuMemcpyHtoD(dl, l, N * sizeof(int))); 
    DRIVER_API_CALL(cuMemcpyHtoD(dr, r, N * sizeof(int))); 

    void *args[6] = {
      &dl, &dr, &dp, &N, &iter1, &iter2
    };

    DRIVER_API_CALL(cuLaunchKernel(vecAdd, blocks, 1, 1, threads, 1, 1, 0, 0, args, 0));

    DRIVER_API_CALL(cuMemcpyDtoH(l, dl, N * sizeof(int))); 
    DRIVER_API_CALL(cuMemcpyDtoH(r, dr, N * sizeof(int))); 
    DRIVER_API_CALL(cuMemcpyDtoH(p, dp, N * sizeof(int))); 
    DRIVER_API_CALL(cuMemFree(dl));
    DRIVER_API_CALL(cuMemFree(dr));
    DRIVER_API_CALL(cuMemFree(dp));

    DRIVER_API_CALL(cuModuleUnload(moduleAdd));

    #pragma omp critical
    {
      printf("Thread %d\n", omp_get_thread_num());
      output(p, N);
    }
  }

  RUNTIME_API_CALL(cudaDeviceSynchronize());
  DRIVER_API_CALL(cuCtxSynchronize());
  DRIVER_API_CALL(cuCtxDestroy(context));

  return 0;
}
