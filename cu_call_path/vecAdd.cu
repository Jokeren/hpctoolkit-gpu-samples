__device__
int __attribute__ ((noinline)) add(int a, int b) {
  return a + b;
}


extern "C"
__global__
void vecAdd(int *l, int *r, int *p, size_t N, size_t iter1, size_t iter2) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (size_t i = 0; i < iter1; ++i) {
    p[idx] = add(l[idx], r[idx]);
  }
  for (size_t i = 0; i < iter2; ++i) {
    p[idx] = add(l[idx], r[idx]);
  }
}
