#include <cstdio>
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#ifdef USE_MPI
#include <mpi.h>
#endif

#include "../utils/common.h"


static size_t N = 1000;


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


__global__
void vecAdd(int *l, int *r, int *p, size_t N) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < N) {
    p[idx] = l[idx] + r[idx];
  }
  cooperative_groups::this_grid().sync();
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
  int device_id = 0;
  if (argc > 1) {
    device_id = atoi(argv[1]);
  }
  cuda_init_device(device_id);

  #pragma omp parallel
  {
    int l[N], r[N], p[N];
    int *dl, *dr, *dp;

    init(l, N);
    init(r, N);

    RUNTIME_API_CALL(cudaMalloc(&dl, N * sizeof(int)));
    RUNTIME_API_CALL(cudaMalloc(&dr, N * sizeof(int)));
    RUNTIME_API_CALL(cudaMalloc(&dp, N * sizeof(int)));

    RUNTIME_API_CALL(cudaMemcpy(dl, l, N * sizeof(int), cudaMemcpyHostToDevice));
    RUNTIME_API_CALL(cudaMemcpy(dr, r, N * sizeof(int), cudaMemcpyHostToDevice));

    dim3 threads(256, 1, 1);
    dim3 blocks((N - 1) / 256 + 1, 1, 1);
    void *args[] = {&dl, &dr, &dp, &N};

    GPU_TEST_FOR((cudaLaunchCooperativeKernel((void*)vecAdd, blocks, threads, args)));

    RUNTIME_API_CALL(cudaMemcpy(p, dp, N * sizeof(int), cudaMemcpyDeviceToHost));

    RUNTIME_API_CALL(cudaFree(dl));
    RUNTIME_API_CALL(cudaFree(dr));
    RUNTIME_API_CALL(cudaFree(dp));

    #pragma omp critical
    {
      printf("Thread %d\n", omp_get_thread_num());
      output(p, N);
    }
  }

  cudaDeviceSynchronize();

#ifdef USE_MPI
  MPI_Finalize();
#endif
  return 0;
}
