#include <cstdio>
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>
#ifdef USE_MPI
#include <mpi.h>
#endif

#include "../utils/common.h"

// XXX: Please set num threads greater than 8
static size_t N = 500000;
static size_t NUM_CONTEXTS = 2;
static size_t NUM_STREAMS_PER_CONTEXT = 4;


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
  int device_id = 0;
  if (argc > 1) {
    device_id = atoi(argv[1]);
  }
  DRIVER_API_CALL(cuInit(0));
  DRIVER_API_CALL(cuDeviceGet(&device, device_id));

  CUcontext contexts[NUM_CONTEXTS];
  CUfunction functions[NUM_CONTEXTS];
  CUstream streams[NUM_STREAMS_PER_CONTEXT * NUM_CONTEXTS];

  for (size_t i = 0; i < NUM_CONTEXTS; ++i) {
    DRIVER_API_CALL(cuCtxCreate(&contexts[i], 0, device));
    DRIVER_API_CALL(cuCtxSetCurrent(contexts[i]));

    for (size_t j = 0; j < NUM_STREAMS_PER_CONTEXT; ++j) {
      DRIVER_API_CALL(cuStreamCreate(&streams[i * NUM_STREAMS_PER_CONTEXT + j], CU_STREAM_NON_BLOCKING));
    }

    CUmodule moduleAdd;
    DRIVER_API_CALL(cuModuleLoad(&moduleAdd, "vecAdd.cubin"));
    DRIVER_API_CALL(cuModuleGetFunction(&functions[i], moduleAdd, "vecAdd"));
  }

  #pragma omp parallel
  {
    size_t thread_id = omp_get_thread_num();
    size_t num_threads = omp_get_num_threads();
    size_t num_streams = NUM_STREAMS_PER_CONTEXT * NUM_CONTEXTS;
    size_t context_id = thread_id / (num_threads / NUM_CONTEXTS);
    size_t stream_id = thread_id / (num_threads / num_streams);
    printf("thread_id %lu context_id %u stream_id %u\n", thread_id, context_id, stream_id);

    CUcontext context = contexts[context_id];
    CUfunction function = functions[context_id];
    CUstream stream = streams[stream_id];

    DRIVER_API_CALL(cuCtxSetCurrent(context));

    int *l = new int[N]();
    int *r = new int[N]();
    int *p = new int[N]();
    CUdeviceptr dl, dr, dp;

    init(l, N);
    init(r, N);

    size_t threads = 256;
    size_t blocks = (N - 1) / threads + 1;


    DRIVER_API_CALL(cuMemAlloc(&dl, N * sizeof(int)));
    DRIVER_API_CALL(cuMemAlloc(&dr, N * sizeof(int)));
    DRIVER_API_CALL(cuMemAlloc(&dp, N * sizeof(int)));
    DRIVER_API_CALL(cuMemcpyHtoD(dl, l, N * sizeof(int))); 
    DRIVER_API_CALL(cuMemcpyHtoD(dr, r, N * sizeof(int))); 

    void *args[4] = {
      &dl, &dr, &dp, &N
    };

    GPU_TEST_FOR(DRIVER_API_CALL(cuLaunchKernel(function, blocks, 1, 1, threads, 1, 1, 0, stream, args, 0)));

    DRIVER_API_CALL(cuMemcpyDtoH(l, dl, N * sizeof(int))); 
    DRIVER_API_CALL(cuMemcpyDtoH(r, dr, N * sizeof(int))); 
    DRIVER_API_CALL(cuMemcpyDtoH(p, dp, N * sizeof(int))); 
    DRIVER_API_CALL(cuMemFree(dl));
    DRIVER_API_CALL(cuMemFree(dr));
    DRIVER_API_CALL(cuMemFree(dp));

    DRIVER_API_CALL(cuStreamSynchronize(stream));
    DRIVER_API_CALL(cuCtxSynchronize());
  }

  for (size_t i = 0; i < NUM_CONTEXTS; ++i) {
    DRIVER_API_CALL(cuCtxDestroy(contexts[i]));
  }

#ifdef USE_MPI
  MPI_Finalize();
#endif
  return 0;
}
