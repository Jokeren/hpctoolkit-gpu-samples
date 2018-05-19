__global__
void vecAdd(float *l, float *r, float *result, size_t N) {
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      result[i * N + j] = l[i] + r[i];
    }
  }
}
