while (error > T && iter < ITER) {
  error = 0.0;

  #pragma omp target data map(A[:H * W], A_new[:H * W])
  {
    #pragma omp target teams map(error)
    #pragma omp distribute parallel for reduction(max:error) collapse(2)
    for (size_t h = 1; h < H - 1; h++) {
      for (size_t w = 1; w < W - 1; w++) {
        A_new[h * W + w] = 0.25 * (A[h * W + w + 1] + A[h * W + w - 1] +
          A[(h + 1) * W + w] + A[(h - 1) * W + w]);
        error = fmax(error, fabs(A_new[h * W + w] - A[h * W + w]));
      } 
    }
  }

  #pragma omp target data map(A[:H * W], A_new[:H * W])
  {
    #pragma omp target teams
    #pragma omp distribute parallel for collapse(2)
    for (size_t h = 1; h < H - 1; h++) {
      for (size_t w = 1; w < W - 1; w++) {
        A[h * W + w] = A_new[h * W + w];
      }
    }
  }

  printf("Phase %lu, error: %f\n", iter, error);
  ++iter;
}
