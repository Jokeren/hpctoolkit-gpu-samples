extern "C"
__global__
void vecSub(int *l, int *r, int *p, size_t N) {
  for (size_t i = 0; i < N; ++i) {
    p[i] = l[i] - r[i];
  }
}
