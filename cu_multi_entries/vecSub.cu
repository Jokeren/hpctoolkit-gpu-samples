extern "C"
__global__
void vecSub(int *l, int *r, int *p, size_t N, size_t iter) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (size_t i = 0; i < iter; ++i) {
    if (idx < N) {
      p[idx] = l[idx] - r[idx];
    }
  }
}
