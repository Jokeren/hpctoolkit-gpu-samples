#include <cstdio>
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>
#ifdef USE_MPI
#include <mpi.h>
#endif

#include "../utils/common.h"


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


#define VEC_ADD(symbol) \
  __global__ \
  void vecAdd_ ##symbol (int *l, int *r, int *p, size_t N) { \
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x; \
    if (idx < N) { \
      p[idx] = l[idx] + r[idx]; \
    } \
  }

VEC_ADD(a)
VEC_ADD(b)
VEC_ADD(c)
VEC_ADD(d)
VEC_ADD(e)
VEC_ADD(f)

void test1(int blocks, int threads, int *l, int *r, int *p) {
  // Input: root->aaa
  // Output: root->aaa
  #pragma nounroll
  for (size_t i = 0; i < 3; ++i) {
    vecAdd_a<<<blocks, threads>>>(l, r, p, N);
  }
}

void test2(int blocks, int threads, int *l, int *r, int *p) {
  // Input: root->aaaa
  // Output: root->AA
  #pragma nounroll
  for (size_t i = 0; i < 4; ++i) {
    vecAdd_a<<<blocks, threads>>>(l, r, p, N);
  }
}

void test3(int blocks, int threads, int *l, int *r, int *p) {
  // Input: root->abccdabccd
  // Output: root->EE
  //         K->abccd
  //        
  //    7. root->AccdA
  //       A->ab
  //
  //    8. root->BcdB
  //       B->abc
  //
  //    9. root->BcdBc
  //       C->abcc
  //
  //    10. root->DdD
  //       D->abcc
  //
  //    11. root->EE
  //       E->abccd
  #pragma nounroll
  for (size_t j = 0; j < 2; ++j) {
    vecAdd_a<<<blocks, threads>>>(l, r, p, N);
    vecAdd_b<<<blocks, threads>>>(l, r, p, N);
    #pragma nounroll
    for (size_t i = 0; i < 2; ++i) {
      vecAdd_c<<<blocks, threads>>>(l, r, p, N);
    }
    vecAdd_d<<<blocks, threads>>>(l, r, p, N);
  }
}

void test4(int blocks, int threads, int *l, int *r, int *p) {
  // Input: root->abcdbcabcd
  // Output: root->CAC
  //         C->aAd
  //         A->bc
  //
  //      5. root->abcdb
  //      6. root->aAdA
  //         A->bc
  //         
  //      7. root->aAdAa
  //      8. root->aAdAab 
  //      9. root->BdAB
  //         B->aA
  //         A->bc
  //      
  //     10. root->CAC
  //         C->aAd
  //         A->bc
  #pragma nounroll
  for (size_t i = 0; i < 3; ++i) {
    if (i == 0 || i == 2) {
      vecAdd_a<<<blocks, threads>>>(l, r, p, N);
    }
    vecAdd_b<<<blocks, threads>>>(l, r, p, N);
    vecAdd_c<<<blocks, threads>>>(l, r, p, N);
    if (i == 0 || i == 2) {
      vecAdd_d<<<blocks, threads>>>(l, r, p, N);
    }
  }
}

void test5(int blocks, int threads, int *l, int *r, int *p) {
  // Input: root->abcdbcabcddbcd
  // Output: root->CACdAd
  //         A->bc
  //         C->aAd
  //
  //        6. root->aAdA
  //           A->bc
  //        7. root->aAdAa
  //        8. root->aAdAab
  //        9. root->BdAB
  //           A->bc
  //           B->aA
  //
  //       10. root->CAC
  //           A->bc
  //           C->aAd
  //
  //        11.root->CACd
  //        12.root->CACdb
  //        13.root->CACdbc
  //        13.root->CACdA
  //        14.root->CACdAd
  #pragma nounroll
  for (size_t i = 0; i < 5; ++i) {
    if (i == 0 || i == 2) {
      vecAdd_a<<<blocks, threads>>>(l, r, p, N);
    }
    if (i == 0 || i == 1 || i == 2 || i == 4) {
      vecAdd_b<<<blocks, threads>>>(l, r, p, N);
    }
    if (i == 0 || i == 1 || i == 2 || i == 4) {
      vecAdd_c<<<blocks, threads>>>(l, r, p, N);
    }
    if (i == 0 || i == 2 || i == 3 || i == 4) {
      vecAdd_d<<<blocks, threads>>>(l, r, p, N);
    }
  }
}

void test6(int blocks, int threads, int *l, int *r, int *p) {
  // Input: root->abcdbcab | (a1)b(a2)d
  // Output: root->aAdAab | (a1)b(a2)d
  //         A->bc
  //
  //       6. root->aAdA
  //          A->bc
  //
  //       7. root->aAdAa
  //       8. root->aAdAab
  //       9. root->aAdAab |
  #pragma nounroll
  for (size_t i = 0; i < 5; ++i) {
    if (i == 0 || i == 2) {
      vecAdd_a<<<blocks, threads>>>(l, r, p, N);
    }
    if (i == 3) {
      vecAdd_a<<<blocks, threads>>>(l, r, p, N);
    }
    if (i == 4) {
      vecAdd_a<<<blocks, threads>>>(l, r, p, N);
    }
    if (i == 0 || i == 1 || i == 2 || i == 3) {
      vecAdd_b<<<blocks, threads>>>(l, r, p, N);
    }
    if (i == 0 || i == 1) {
      vecAdd_c<<<blocks, threads>>>(l, r, p, N);
    }
    if (i == 0 || i == 4) {
      vecAdd_d<<<blocks, threads>>>(l, r, p, N);
    }
  }
}

void test7(int blocks, int threads, int *l, int *r, int *p) {
  // Input: root->abcdbcab | (a1)ab
  // Output: root->aAdAB | (a1)B
  //         A->bc
  //         B->ab
  #pragma nounroll
  for (size_t i = 0; i < 5; ++i) {
    if (i == 0 || i == 2 || i == 4) {
      vecAdd_a<<<blocks, threads>>>(l, r, p, N);
    }
    if (i == 3) {
      vecAdd_a<<<blocks, threads>>>(l, r, p, N);
    }
    if (i == 0 || i == 1 || i == 2 || i == 4) {
      vecAdd_b<<<blocks, threads>>>(l, r, p, N);
    }
    if (i == 0 || i == 1) {
      vecAdd_c<<<blocks, threads>>>(l, r, p, N);
    }
    if (i == 0) {
      vecAdd_d<<<blocks, threads>>>(l, r, p, N);
    }
  }
}

void test8(int blocks, int threads, int *l, int *r, int *p) {
  // Input: root-> abc | (a1) | abc
  // Output: root-> | B | (a1) | B
  #pragma nounroll
  for (size_t i = 0; i < 3; ++i) {
    if (i == 0 || i == 2) {
      vecAdd_a<<<blocks, threads>>>(l, r, p, N);
    }
    if (i == 1) {
      vecAdd_a<<<blocks, threads>>>(l, r, p, N);
    }
    if (i == 0 || i == 2) {
      vecAdd_b<<<blocks, threads>>>(l, r, p, N);
    }
    if (i == 0 || i == 2) {
      vecAdd_c<<<blocks, threads>>>(l, r, p, N);
    }
  }
}

void test9(int blocks, int threads, int *l, int *r, int *p) {
  // Input: root-> abc | (a1) | abc | (a1)
  // Output: root-> | B | (a1) | (a1)
  #pragma nounroll
  for (size_t i = 0; i < 4; ++i) {
    if (i == 0 || i == 2) {
      vecAdd_a<<<blocks, threads>>>(l, r, p, N);
    }
    if (i == 1 || i == 3) {
      vecAdd_a<<<blocks, threads>>>(l, r, p, N);
    }
    if (i == 0 || i == 2) {
      vecAdd_b<<<blocks, threads>>>(l, r, p, N);
    }
    if (i == 0 || i == 2) {
      vecAdd_c<<<blocks, threads>>>(l, r, p, N);
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

  int test_id = 1;
  if (argc > 2) {
    test_id = atoi(argv[2]);
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
    
    if (test_id == 1) {
      test1(blocks, threads, dl, dr, dp);
    } else if (test_id == 2) {
      test2(blocks, threads, dl, dr, dp);
    } else if (test_id == 3) {
      test3(blocks, threads, dl, dr, dp);
    } else if (test_id == 4) {
      test4(blocks, threads, dl, dr, dp);
    } else if (test_id == 5) {
      test5(blocks, threads, dl, dr, dp);
    } else if (test_id == 6) {
      test6(blocks, threads, dl, dr, dp);
    } else if (test_id == 7) {
      test7(blocks, threads, dl, dr, dp);
    } else if (test_id == 8) {
      test8(blocks, threads, dl, dr, dp);
    } else if (test_id == 9) {
      test9(blocks, threads, dl, dr, dp);
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
