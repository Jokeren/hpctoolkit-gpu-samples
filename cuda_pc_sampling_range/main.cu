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

__global__
void vecAdd3(int *l, int *r, int *p, size_t N) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < N) {
    p[idx] = l[idx] - r[idx];
  }
}

__global__
void vecAdd4(int *l, int *r, int *p, size_t N) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < N) {
    p[idx] = l[idx] * r[idx];
  }
}

__global__
void vecAdd5(int *l, int *r, int *p, size_t N) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < N) {
    p[idx] = l[idx] / r[idx];
  }
}


void test1(int *dl, int *dr, int *dp, int threads, int blocks) {
  #pragma loop nounroll
  for (size_t i = 0; i < 1 << 12; ++i) {
    // C1
    GPU_TEST_FOR((vecAdd1<<<blocks, threads>>>(dl, dr, dp, N)));

    // C2
    GPU_TEST_FOR((vecAdd1<<<blocks, threads>>>(dl, dr, dp, N)));
  }
}


void test2(int *dl, int *dr, int *dp, int threads, int blocks) {
  #pragma loop nounroll
  for (size_t i = 0; i < 1 << 12; ++i) {
    // C1
    GPU_TEST_FOR((vecAdd1<<<blocks, threads>>>(dl, dr, dp, N)));

    // C2
    GPU_TEST_FOR((vecAdd1<<<blocks, threads>>>(dl, dr, dp, N / 2)));
  }
}


void test3(int *dl, int *dr, int *dp, int threads, int blocks) {
  #pragma loop nounroll
  for (size_t i = 0; i < 1 << 12; ++i) {
    // C1
    GPU_TEST_FOR((vecAdd2<<<blocks, threads>>>(dl, dr, dp, N)));

    // C2
    GPU_TEST_FOR((vecAdd2<<<blocks, threads>>>(dl, dr, dp, N / 2)));
  }
}

void test4(int *dl, int *dr, int *dp, int threads, int blocks) {
  #pragma loop nounroll
  for (size_t i = 0; i < 1 << 12; ++i) {
    // C1
    GPU_TEST_FOR((vecAdd1<<<blocks, threads>>>(dl, dr, dp, N)));

    // C2
    GPU_TEST_FOR((vecAdd2<<<blocks, threads>>>(dl, dr, dp, N / 2)));

    if (i % 3 == 0) {
      // C3
      GPU_TEST_FOR((vecAdd3<<<blocks, threads>>>(dl, dr, dp, N / 4)));
    } else if (i % 3 == 1) {
      // C4
      GPU_TEST_FOR((vecAdd4<<<blocks, threads>>>(dl, dr, dp, N / 8)));
    } else {
      // C5
      GPU_TEST_FOR((vecAdd5<<<blocks, threads>>>(dl, dr, dp, N / 8)));
    }

    GPU_TEST_FOR((vecAdd1<<<blocks, threads>>>(dl, dr, dp, N)));
  }
}


void test5(int *dl, int *dr, int *dp, int threads, int blocks) {
  #pragma loop nounroll
  for (size_t i = 0; i < 1 << 10; ++i) {
    // C1
    GPU_TEST_FOR((vecAdd1<<<blocks, threads>>>(dl, dr, dp, N)));

    // C2
    GPU_TEST_FOR((vecAdd2<<<blocks, threads>>>(dl, dr, dp, N / 2)));

    // C3
    GPU_TEST_FOR((vecAdd3<<<blocks, threads>>>(dl, dr, dp, N / 4)));
  }
}


void test6(int *dl, int *dr, int *dp, int threads, int blocks) {
  #pragma loop nounroll
  for (size_t j = 0; j < 2; ++j) {
    #pragma loop nounroll
    for (size_t i = 0; i < 1 << 10; ++i) {
      // C1
      GPU_TEST_FOR((vecAdd1<<<blocks, threads>>>(dl, dr, dp, N)));

      // C2
      GPU_TEST_FOR((vecAdd2<<<blocks, threads>>>(dl, dr, dp, N / 2)));

      // C3
      if (j == 1 && i > (1 << 5)) {
        GPU_TEST_FOR((vecAdd4<<<blocks, threads>>>(dl, dr, dp, N / 4)));
      } else {
        GPU_TEST_FOR((vecAdd3<<<blocks, threads>>>(dl, dr, dp, N / 4)));
      }
    }

    GPU_TEST_FOR((vecAdd1<<<blocks, threads>>>(dl, dr, dp, N)));
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

  int mode = 1;
  if (argc > 2) {
    mode = atoi(argv[2]);
  }

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

    if (mode == 1) {
      // Test case 1
      // C2 should the same number samples as C1
      test1(dl, dr, dp, threads, blocks);
    } else if (mode == 2) {
      // Test case 2
      // C2 should get half the samples of C1
      // The equal range mode should fail in this case
      test2(dl, dr, dp, threads, blocks);
    } else if (mode == 3) {
      // Test case 3
      // C2's add should be half of C1's add
      // The equal range mode should fail in this case
      test3(dl, dr, dp, threads, blocks);
    } else if (mode == 4) {
      // Test case 4
      // Test range ids
      test4(dl, dr, dp, threads, blocks);
    } else if (mode == 5) {
      // Test case 5
      // Test compress rate
      test5(dl, dr, dp, threads, blocks);
    } else if (mode == 6) {
      // Test case 5
      // Test split function
      test6(dl, dr, dp, threads, blocks);
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
