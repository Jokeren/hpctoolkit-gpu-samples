__global__
void vecAdd(float *l, float *r, float *result, size_t N) {
  for (size_t i = 0; i < N; ++i) {
    result[i] = l[i] + r[i];
  }
}
