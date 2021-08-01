#include <cstdio>
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pthread.h>
#ifdef USE_MPI
#include <mpi.h>
#endif

#include "../utils/common.h"

static size_t N = 10000;
static size_t NUM_CONTEXTS = 1;
static size_t NUM_STREAMS_PER_CONTEXT = 1;
static size_t NUM_THREADS = 1;
static size_t DEPTH = 0;
static size_t DEVICE_ID = 0;
static bool DEBUG = false;

void setup() {
  char *buf = NULL;

  if ((buf = getenv("BENCH_NUM_ELEMENTS")) != NULL) {
    N = atoi(buf);
  }

  if ((buf = getenv("BENCH_NUM_THREADS")) != NULL) {
    NUM_THREADS = atoi(buf);
  }

  if ((buf = getenv("BENCH_NUM_CONTEXTS")) != NULL) {
    NUM_CONTEXTS = atoi(buf);
  }

  if ((buf = getenv("BENCH_NUM_STREAMS_PER_CONTEXT")) != NULL) {
    NUM_STREAMS_PER_CONTEXT = atoi(buf);
  }

  if ((buf = getenv("BENCH_DEPTH")) != NULL) {
    DEPTH = atoi(buf);
  }

  if ((buf = getenv("BENCH_DEVICE_ID")) != NULL) {
    DEVICE_ID = atoi(buf);
  }

  if ((buf = getenv("BENCH_DEBUG")) != NULL) {
    DEBUG = atoi(buf);
  }
}

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

void launch_kernel(size_t cur_depth, CUfunction function, size_t blocks, size_t threads,
  CUstream stream, void *args[4]) {
  if (cur_depth < DEPTH) {
    launch_kernel(++cur_depth, function, blocks, threads, stream, args);
  } else if (cur_depth == DEPTH) {
    GPU_TEST_FOR(DRIVER_API_CALL(cuLaunchKernel(function, blocks, 1, 1, threads, 1, 1, 0, stream, args, 0)));
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

  setup();

  // Init device
  CUdevice device;
  DRIVER_API_CALL(cuInit(0));
  DRIVER_API_CALL(cuDeviceGet(&device, DEVICE_ID));

  CUcontext contexts[NUM_CONTEXTS];
  CUfunction functions[NUM_CONTEXTS];
  CUmodule moduleAdd[NUM_CONTEXTS];
  CUstream streams[NUM_STREAMS_PER_CONTEXT * NUM_CONTEXTS];

  for (size_t i = 0; i < NUM_CONTEXTS; ++i) {
    DRIVER_API_CALL(cuCtxCreate(&contexts[i], 0, device));
    DRIVER_API_CALL(cuCtxSetCurrent(contexts[i]));

    if (DEBUG) {
      printf("context: %p\n", contexts[i]);
    }

    for (size_t j = 0; j < NUM_STREAMS_PER_CONTEXT; ++j) {
      DRIVER_API_CALL(cuStreamCreate(&streams[i * NUM_STREAMS_PER_CONTEXT + j], CU_STREAM_NON_BLOCKING));
    }

    DRIVER_API_CALL(cuModuleLoad(&moduleAdd[i], "vecAdd.cubin"));
    DRIVER_API_CALL(cuModuleGetFunction(&functions[i], moduleAdd[i], "vecAdd"));

    DRIVER_API_CALL(cuCtxSetCurrent(NULL));
  }

  omp_set_num_threads(NUM_THREADS);

#pragma omp parallel
  {
    size_t thread_id = omp_get_thread_num();
    size_t num_threads = omp_get_num_threads();
    size_t num_streams = NUM_STREAMS_PER_CONTEXT * NUM_CONTEXTS;
    size_t context_id = thread_id / (num_threads / NUM_CONTEXTS);
    size_t stream_id = thread_id / (num_threads / num_streams);

    if (DEBUG) {
      printf("thread_id %lu context_id %u stream_id %u\n", thread_id, context_id, stream_id);
    }

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

    DRIVER_API_CALL(cuMemcpyHtoDAsync(dl, l, N * sizeof(int), stream));
    DRIVER_API_CALL(cuMemcpyHtoDAsync(dr, r, N * sizeof(int), stream));

    void *args[4] = {
      &dl, &dr, &dp, &N
    };

    launch_kernel(0, function, blocks, threads, stream, args);

    DRIVER_API_CALL(cuMemcpyDtoHAsync(l, dl, N * sizeof(int), stream));
    DRIVER_API_CALL(cuMemcpyDtoHAsync(r, dr, N * sizeof(int), stream));
    DRIVER_API_CALL(cuMemcpyDtoHAsync(p, dp, N * sizeof(int), stream));

    DRIVER_API_CALL(cuStreamSynchronize(stream));

    DRIVER_API_CALL(cuMemFree(dl));
    DRIVER_API_CALL(cuMemFree(dr));
    DRIVER_API_CALL(cuMemFree(dp));

    DRIVER_API_CALL(cuCtxSynchronize());

    delete [] l;
    delete [] r;
    delete [] p;

    DRIVER_API_CALL(cuCtxSetCurrent(NULL));

    if (DEBUG) {
      #pragma omp critical
      output(p, N);
    }
  }

  for (size_t i = 0; i < NUM_CONTEXTS; ++i) {
    DRIVER_API_CALL(cuCtxSetCurrent(contexts[i]));

    for (size_t j = 0; j < NUM_STREAMS_PER_CONTEXT; ++j) {
      DRIVER_API_CALL(cuStreamDestroy(streams[i * NUM_STREAMS_PER_CONTEXT + j]));
    }

    DRIVER_API_CALL(cuModuleUnload(moduleAdd[i]));
    DRIVER_API_CALL(cuCtxSetCurrent(NULL));
    DRIVER_API_CALL(cuCtxDestroy(contexts[i]));
  }

#ifdef USE_MPI
  MPI_Finalize();
#endif
  return 0;
}
