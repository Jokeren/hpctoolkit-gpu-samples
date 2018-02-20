#include <cstdio>
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>
#ifdef USE_MPI
#include <mpi.h>
#endif

static const size_t N = 1000;

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
}

int main(int argc, char *argv[]) {
  int l1[N], l2[N];
  int r1[N], r2[N];
  int p1[N], p2[N];
  int *dl1, *dl2;
  int *dr1, *dr2;
  int *dp1, *dp2;

  init(l1, N);
  init(r1, N);
  init(l2, N);
  init(r2, N);

  cudaMalloc(&dl1, N * sizeof(int));
  cudaMalloc(&dl2, N * sizeof(int));
  cudaMalloc(&dr1, N * sizeof(int));
  cudaMalloc(&dr2, N * sizeof(int));
  cudaMalloc(&dp1, N * sizeof(int));
  cudaMalloc(&dp2, N * sizeof(int));

  cudaMemcpy(dl1, l1, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dl2, l2, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dr1, r1, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dr2, r2, N * sizeof(int), cudaMemcpyHostToDevice);

#ifdef USE_MPI
  int numtasks, rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  printf("MPI task %d/%d\n", rank, numtasks);
#endif

  #pragma omp parallel
  {
    size_t threads = 256;
    size_t blocks = (N - 1) / threads + 1;
    if (omp_get_thread_num() == 0) {
      vecAdd<<<blocks, threads>>>(dl1, dr1, dp1, N); 
    } else if (omp_get_thread_num() == 1) {
      vecAdd<<<blocks, threads>>>(dl2, dr2, dp2, N); 
    }
  }

  cudaMemcpy(p1, dp1, N * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(p2, dp2, N * sizeof(int), cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  output(p1, N);
  output(p2, N);

  cudaFree(dl1);
  cudaFree(dl2);
  cudaFree(dr1);
  cudaFree(dr2);
  cudaFree(dp1);
  cudaFree(dp2);
#ifdef USE_MPI
  MPI_Finalize();
#endif
  return 0;
}
