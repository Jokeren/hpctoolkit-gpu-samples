__device__
int __attribute__ ((noinline)) add(int *l, int *r, int i, int N) {
  if (i < N) {
    return l[i] + r[i] + add(l, r, N, N);
  } else {
    return l[0] + r[0];
  }
}

extern "C"
__global__
void vecAdd(int *l, int *r, int *p, size_t N, size_t iter) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (size_t i = 0; i < iter; ++i) {
    p[idx] = add(l, r, idx, N);
  }
}
