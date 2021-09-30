#include <cstdio>
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>
#ifdef USE_MPI
#include <mpi.h>
#endif

#include "../utils/common.h"


static const size_t N = 10000;


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
void vecAdd1(int *l, int *r, int *p, size_t N) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < N) {
    p[idx] = l[idx] + r[idx];
  }
}

__device__
int __attribute__ ((noinline)) add(int l, int r) {
  return l + r;
}


__global__
void vecAdd2(int *l, int *r, int *p, size_t N) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < N) {
    p[idx] = add(l[idx], r[idx]);
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

    size_t threads = 256;
    size_t blocks = (N - 1) / threads + 1;

    // Test case 1
    // C2 should the same number samples as C1
    #pragma loop nounroll
    for (size_t i = 0; i < 128; ++i) {
      // C1
      GPU_TEST_FOR((vecAdd1<<<blocks, threads>>>(dl, dr, dp, N)));

      // C2
      GPU_TEST_FOR((vecAdd1<<<blocks, threads>>>(dl, dr, dp, N)));
    }

    // Test case 2
    // C2 should get half the samples of C1
    // The equal range mode should fail in this case
    #pragma loop nounroll
    for (size_t i = 0; i < 128; ++i) {
      // C1
      GPU_TEST_FOR((vecAdd1<<<blocks, threads>>>(dl, dr, dp, N)));

      // C2
      GPU_TEST_FOR((vecAdd1<<<blocks, threads>>>(dl, dr, dp, N / 2)));
    }

    // Test case 3
    // C2's add should be half of C1's add
    // The equal range mode should fail in this case
    #pragma loop nounroll
    for (size_t i = 0; i < 128; ++i) {
      // C1
      GPU_TEST_FOR((vecAdd2<<<blocks, threads>>>(dl, dr, dp, N)));

      // C2
      GPU_TEST_FOR((vecAdd2<<<blocks, threads>>>(dl, dr, dp, N / 2)));
    }

    RUNTIME_API_CALL(cudaMemcpy(p, dp, N * sizeof(int), cudaMemcpyDeviceToHost));

    RUNTIME_API_CALL(cudaFree(dl));
    RUNTIME_API_CALL(cudaFree(dr));
    RUNTIME_API_CALL(cudaFree(dp));

#ifdef OUTPUT
    #pragma omp critical
    {
      printf("Thread %d\n", omp_get_thread_num());
      output(p, N);
    }
#endif
  }

  cudaDeviceSynchronize();

#ifdef USE_MPI
  MPI_Finalize();
#endif
  return 0;
}
