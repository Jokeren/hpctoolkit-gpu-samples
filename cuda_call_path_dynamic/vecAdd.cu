#include "vecAdd.h"

__global__
void add(int *l, int *r, int *p, size_t i) {
  p[i] = l[i] + r[i];
}

__global__
void vecAdd(int *l, int *r, int *p, size_t i, size_t N) {
  if (i < N) {
    p[i] = l[i] + r[i];
    __syncthreads();
    if (threadIdx.x == 0) {
      add<<<1, 32>>>(l, r, p, i);
    }
  }
}
