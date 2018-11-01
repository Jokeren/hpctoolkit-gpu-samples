__device__ int add_proxy(int *l, int *r, int i, int N);
__device__ int mul_proxy1(int *l, int *r, int i, int N);
__device__ int mul_proxy2(int *l, int *r, int i, int N);


__device__
int __attribute__ ((noinline)) add(int *l, int *r, int i, int N) {
  if (i < N) {
    return l[i] + r[i] + add_proxy(l, r, N, N) + mul_proxy1(l, r, i, N);
  } else {
    return l[0] + r[0];
  }
}


__device__
int __attribute__ ((noinline)) add_proxy(int *l, int *r, int i, int N) {
  return add(l, r, i, N);
}


__device__
int __attribute__ ((noinline)) mul_proxy1(int *l, int *r, int i, int N) {
  if (i < N) {
    return mul_proxy2(l, r, i, N);
  } else {
    return l[0] * r[0];
  }
}


__device__
int __attribute__ ((noinline)) mul_proxy2(int *l, int *r, int i, int N) {
  return mul_proxy1(l, r, N, N);
}


extern "C"
__global__
void vecAdd(int *l, int *r, int *p, size_t N, size_t iter) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (size_t i = 0; i < iter; ++i) {
    if (idx < N) {
      p[idx] = add(l, r, idx, N);
    }
  }
}
