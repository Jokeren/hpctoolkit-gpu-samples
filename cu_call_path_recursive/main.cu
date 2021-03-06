#include <cstdio>
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>
#ifdef USE_MPI
#include <mpi.h>
#endif

#include "../utils/common.h"


static size_t N = 20;
static size_t iter = 100;


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
#ifdef USE_MPI
  int numtasks, rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  printf("MPI task %d/%d\n", rank, numtasks);
#endif

  // Init device
  CUdevice device;
  CUcontext context;
  CUmodule module;
  CUfunction function;
  int device_id = 0;
  if (argc > 1) {
    device_id = atoi(argv[1]);
  }
  cu_init_device(device_id, device, context);
  cu_load_module_function(module, "vecAdd.cubin", function, "vecAdd");

  #pragma omp parallel
  {
    int l[N], r[N], p[N];
    CUdeviceptr dl, dr, dp;

    init(l, N);
    init(r, N);

    size_t threads = 256;
    size_t blocks = (N - 1) / threads + 1;

    DRIVER_API_CALL(cuCtxSetCurrent(context));

    DRIVER_API_CALL(cuMemAlloc(&dl, N * sizeof(int)));
    DRIVER_API_CALL(cuMemAlloc(&dr, N * sizeof(int)));
    DRIVER_API_CALL(cuMemAlloc(&dp, N * sizeof(int)));
    DRIVER_API_CALL(cuMemcpyHtoD(dl, l, N * sizeof(int))); 
    DRIVER_API_CALL(cuMemcpyHtoD(dr, r, N * sizeof(int))); 

    void *args[5] = {
      &dl, &dr, &dp, &N, &iter
    };

    GPU_TEST_FOR(DRIVER_API_CALL(cuLaunchKernel(function, blocks, 1, 1, threads, 1, 1, 0, 0, args, 0)));

    DRIVER_API_CALL(cuMemcpyDtoH(l, dl, N * sizeof(int))); 
    DRIVER_API_CALL(cuMemcpyDtoH(r, dr, N * sizeof(int))); 
    DRIVER_API_CALL(cuMemcpyDtoH(p, dp, N * sizeof(int))); 
    DRIVER_API_CALL(cuMemFree(dl));
    DRIVER_API_CALL(cuMemFree(dr));
    DRIVER_API_CALL(cuMemFree(dp));

#ifdef OUTPUT
    #pragma omp critical
    {
      printf("Thread %d\n", omp_get_thread_num());
      output(p, N);
    }
#endif

    DRIVER_API_CALL(cuCtxSynchronize());
  }

  DRIVER_API_CALL(cuModuleUnload(module));
  DRIVER_API_CALL(cuCtxDestroy(context));
  RUNTIME_API_CALL(cudaDeviceSynchronize());

#ifdef USE_MPI
  MPI_Finalize();
#endif
  return 0;
}
