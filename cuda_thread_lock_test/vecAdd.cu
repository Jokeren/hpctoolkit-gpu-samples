extern "C"
__global__
void vecAdd(int *l, int *r, int *p, size_t N, size_t iter) {
  size_t idx = threadIdx.x;
  if (idx < N) {
    p[idx] = l[idx] + r[idx];
  }
}
