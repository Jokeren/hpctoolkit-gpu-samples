__global__
void vecAdd(float *l, float *r, float *result, size_t N) {
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      for (size_t k = 0; k < N; ++k) {
        for (size_t v = 0; v < N; ++v) {
          for (size_t u = 0; u < N; ++u) {
            result[i + j + k + v + u] = l[i + j] + r[v + u + k];
          }
        }
      }
    }
  }
}

