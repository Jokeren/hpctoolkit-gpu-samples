#include <cstdio>
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>
#ifdef USE_MPI
#include <mpi.h>
#endif

#include "../utils/common.h"


static const size_t N = 2000;
static const size_t ITER = 1000;


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
void vecAdd1(int *l, int *r, int *p, size_t ITER) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  __shared__ int tile1[1536];
  __shared__ int tile2[1536];
  tile1[idx] = l[idx];
  tile2[idx] = r[idx];
  p[idx] = 0;
  __syncthreads();
  for (size_t i = 0; i < ITER; ++i) {
    for (size_t j = 0; j < 128; ++j) {
      p[idx] += tile1[(idx % 32) * 32 + j] + tile2[(idx % 32) * 32 + j];
    }
  }
}


__global__
void vecAdd2(int *l, int *r, int *p, size_t ITER) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  __shared__ int tile1[1536];
  __shared__ int tile2[1536];
  tile1[idx] = l[idx];
  tile2[idx] = r[idx];
  p[idx] = 0;
  __syncthreads();
  for (size_t i = 0; i < ITER; ++i) {
    for (size_t j = 0; j < 128; ++j) {
      p[idx] += tile1[idx / 32 + j] + tile2[idx / 32 + j];
    }
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

    size_t threads = 512;
    size_t blocks = 1;

    GPU_TEST_FOR((vecAdd1<<<blocks, threads>>>(dl, dr, dp, ITER)));

    GPU_TEST_FOR((vecAdd2<<<blocks, threads>>>(dl, dr, dp, ITER)));

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
