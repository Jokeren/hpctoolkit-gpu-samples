while (error > T && iter < ITER) {
  error = 0.0;

  #pragma omp parallel for reduction(max:error)
  for (size_t h = 1; h < H - 1; h++) {
    for (size_t w = 1; w < W - 1; w++) {
      A_new[h * W + w] = 0.25 * (A[h * W + w + 1] + A[h * W + w - 1] +
        A[(h + 1) * W + w] + A[(h - 1) * W + w]);
      error = std::max(error, std::abs(A_new[h * W + w] - A[h * W + w]));
    } 
  }

  #pragma omp parallel for
  for (size_t h = 1; h < H - 1; h++) {
    for (size_t w = 1; w < W - 1; w++) {
      A[h * W + w] = A_new[h * W + w];
    }
  }

  printf("Phase %lu, error: %f\n", iter, error);
  ++iter;
}
