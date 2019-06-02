__global__
void nekbone(double *w, double *u, double *g, double *d, double *dt, const int N) {
  const int e_size = N * N * N;
  const int e_offset = e_size * blockIdx.x;

  __shared__ double ur[1024];
  __shared__ double us[1024];
  __shared__ double ut[1024];
  __shared__ double ul[1024];

  for (int it = threadIdx.x; it < e_size; it += blockDim.x) {
    ul[it] = u[e_offset + it];
  }

  __syncthreads();

  int i, j, k;
  for (int it = threadIdx.x; it < e_size; it += blockDim.x) {
    j = it / N;
    i = it - j * N;
    k = j / N;
    j -= k * N;
    double wr = 0.0;
    double ws = 0.0;
    double wt = 0.0;
    for (int n = 0; n < N; ++n) {
      wr += dt[i * N + n] * ul[N * (j + k * N) + n];
      ws += dt[j * N + n] * ul[N * (n + k * N) + i];
      wt += dt[k * N + n] * ul[N * (j + n * N) + i];
    }
    int g_offset = 6 * (e_offset + it);
    ur[it] = g[g_offset + 0] * wr + g[g_offset + 1] * ws + g[g_offset + 2] * wt;
    us[it] = g[g_offset + 1] * wr + g[g_offset + 3] * ws + g[g_offset + 4] * wt;
    ut[it] = g[g_offset + 2] * wr + g[g_offset + 4] * ws + g[g_offset + 5] * wt;
  }

  __syncthreads();

  for (int it = threadIdx.x; it < e_size; it += blockDim.x) {
    j = it / N;
    i = it - j * N;
    k = j / N;
    j -= k * N;
    double s = 0.0;
    for (int n = 0; n < N; ++n) {
      s += d[i * N + n] * ur[N * (j + N * k) + n] +
        d[j * N + n] * us[N * (n + N * k) + i] +
        d[k * N + n] * ut[N * (j + N * n) + i];
    }
    w[e_offset + it] = s;
  }
}

