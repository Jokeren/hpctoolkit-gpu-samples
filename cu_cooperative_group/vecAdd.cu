#include <cooperative_groups.h>

extern "C"
__global__
void vecAdd(int *l, int *r, int *p, size_t N) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < N) {
    p[idx] = l[idx] + r[idx];
  }
  cooperative_groups::this_grid().sync();
}
