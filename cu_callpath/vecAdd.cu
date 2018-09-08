__device__
int __attribute__ ((noinline)) add(int a, int b) {
  return a + b;
}


extern "C"
__global__
void vecAdd(int *l, int *r, int *p, size_t N, size_t iter) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (size_t i = 0; i < iter; ++i) {
    if (idx < N) {
      p[idx] = add(l[idx], r[idx]);
    }
  }
}
