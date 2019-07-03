__device__ void add_proxy1(int *l, int *r, int *p, size_t N, size_t iter1, size_t iter2);
__device__ void add_proxy2(int *l, int *r, int *p, size_t N, size_t iter1, size_t iter2);
__device__ void add_proxy3(int *l, int *r, int *p, size_t N, size_t iter1, size_t iter2);
__device__ int add(int a, int b);


__device__ void __attribute__ ((noinline))
add_proxy1(int *l, int *r, int *p, size_t N, size_t iter1, size_t iter2) {
  add_proxy2(l, r, p, N, iter1, iter2);
}


__device__ void __attribute__ ((noinline))
add_proxy2(int *l, int *r, int *p, size_t N, size_t iter1, size_t iter2) {
  add_proxy3(l, r, p, N, iter1, iter2);
}


__device__ void __attribute__ ((noinline))
add_proxy3(int *l, int *r, int *p, size_t N, size_t iter1, size_t iter2) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (size_t i = 0; i < iter1; ++i) {
    if (idx < N) {
      p[idx] = add(l[idx], r[idx]);
    }
  }
  for (size_t i = 0; i < iter2; ++i) {
    if (idx < N) {
      p[idx] = add(l[idx], r[idx]);
    }
  }
}


__device__ int add(int a, int b) {
  return a + b;
}


extern "C"
__global__
void vecAdd(int *l, int *r, int *p, size_t N, size_t iter1, size_t iter2) {
  add_proxy1(l, r, p, N, iter1, iter2);
}
