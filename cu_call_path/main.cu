#include <cstdio>
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>
#ifdef USE_MPI
#include <mpi.h>
#endif

#include "../utils/common.h"


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
  int l1[N], l2[N];
  int r1[N], r2[N];
  int p1[N], p2[N];
  CUdeviceptr dl1, dl2;
  CUdeviceptr dr1, dr2;
  CUdeviceptr dp1, dp2;

  init(l1, N);
  init(r1, N);
  init(l2, N);
  init(r2, N);

#ifdef USE_MPI
  int numtasks, rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  printf("MPI task %d/%d\n", rank, numtasks);
#endif

  CUdevice device;
  CUcontext context;
  int device_id = 0;
  if (argc > 1) {
    device_id = atoi(argv[1]);
  }
  init_device(device_id, device, context);

  #pragma omp parallel
  {
    size_t threads = 256;
    size_t blocks = (N - 1) / threads + 1;
    DRIVER_API_CALL(cuCtxSetCurrent(context));
    CUmodule moduleAdd;
    CUfunction vecAdd;

    DRIVER_API_CALL(cuModuleLoad(&moduleAdd, "vecAdd.cubin"));
    DRIVER_API_CALL(cuModuleGetFunction(&vecAdd, moduleAdd, "vecAdd"));

    if (omp_get_thread_num() == 0) {
      DRIVER_API_CALL(cuMemAlloc(&dl1, N * sizeof(int)));
      DRIVER_API_CALL(cuMemAlloc(&dr1, N * sizeof(int)));
      DRIVER_API_CALL(cuMemAlloc(&dp1, N * sizeof(int)));
      DRIVER_API_CALL(cuMemcpyHtoD(dl1, l1, N * sizeof(int))); 
      DRIVER_API_CALL(cuMemcpyHtoD(dr1, r1, N * sizeof(int))); 

      void *args[6] = {
        &dl1, &dr1, &dp1, &N, &iter1, &iter2
      };

      DRIVER_API_CALL(cuLaunchKernel(vecAdd, blocks, 1, 1, threads, 1, 1, 0, 0, args, 0));

      DRIVER_API_CALL(cuMemcpyDtoH(l1, dl1, N * sizeof(int))); 
      DRIVER_API_CALL(cuMemcpyDtoH(r1, dr1, N * sizeof(int))); 
      DRIVER_API_CALL(cuMemcpyDtoH(p1, dp1, N * sizeof(int))); 
      DRIVER_API_CALL(cuMemFree(dl1));
      DRIVER_API_CALL(cuMemFree(dr1));
      DRIVER_API_CALL(cuMemFree(dp1));
    } else if (omp_get_thread_num() == 1) {
      DRIVER_API_CALL(cuMemAlloc(&dl2, N * sizeof(int)));
      DRIVER_API_CALL(cuMemAlloc(&dr2, N * sizeof(int)));
      DRIVER_API_CALL(cuMemAlloc(&dp2, N * sizeof(int)));
      DRIVER_API_CALL(cuMemcpyHtoD(dl2, l2, N * sizeof(int))); 
      DRIVER_API_CALL(cuMemcpyHtoD(dr2, r2, N * sizeof(int))); 

      void *args[6] = {
        &dl2, &dr2, &dp2, &N, &iter1, &iter2
      };

      DRIVER_API_CALL(cuLaunchKernel(vecAdd, blocks, 1, 1, threads, 1, 1, 0, 0, args, 0));

      DRIVER_API_CALL(cuMemcpyDtoH(l2, dl2, N * sizeof(int))); 
      DRIVER_API_CALL(cuMemcpyDtoH(r2, dr2, N * sizeof(int))); 
      DRIVER_API_CALL(cuMemcpyDtoH(p2, dp2, N * sizeof(int))); 
      DRIVER_API_CALL(cuMemFree(dl2));
      DRIVER_API_CALL(cuMemFree(dr2));
      DRIVER_API_CALL(cuMemFree(dp2));
    }
    DRIVER_API_CALL(cuModuleUnload(moduleAdd));
  }

  cudaDeviceSynchronize();
  DRIVER_API_CALL(cuCtxSynchronize());
  DRIVER_API_CALL(cuCtxDestroy(context));

  output(p1, N);
  output(p2, N);

#ifdef USE_MPI
  MPI_Finalize();
#endif
  return 0;
}
