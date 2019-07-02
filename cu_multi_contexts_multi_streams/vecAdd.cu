extern "C"
__global__
void vecAdd(int *l, int *r, int *p, size_t N) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < N) {
    for (size_t i = 0; i < 1000; ++i) {
      p[idx] = l[idx] + r[idx];
    }
  }
}
