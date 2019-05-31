__global__
void nekbone(double *w, double *u, double *g, double *d, const int N) {
  const int e_size = N * N * N;
  const int e_offset = e_size * blockIdx.x;

  __shared__ double ur[1024];
  __shared__ double us[1024];
  __shared__ double ut[1024];
  __shared__ double ul[1024];
  __shared__ double d_s[128];

  for (int it = threadIdx.x ; it < e_size; it += blockDim.x) {
    ul[it] = u[e_offset + it];
  }
  
  if (threadIdx.x < 128) {
    d_s[threadIdx.x] = d[threadIdx.x];
  }

  __syncthreads();

  int i, j, k;
  for(int it = threadIdx.x; it < e_size; it += blockDim.x) {
    j = it / N;
    i = it - j * N;
    k = j / N;
    j -= k * N;
    double wr = 0.0;
    double ws = 0.0;
    double wt = 0.0;
    for(int n = 0; n < N; ++n) {
      wr += d_s[n * N + i] * ul[N * (j + k * N) + n];
      ws += d_s[n * N + j] * ul[N * (n + k * N) + i];
      wt += d_s[n * N + k] * ul[N * (j + n * N) + i];
    }
    int g_offset = 6 * (e_offset + it);
    ur[it] = g[g_offset + 0] * wr + g[g_offset + 1] * ws + g[g_offset + 2] * wt;
    us[it] = g[g_offset + 1] * wr + g[g_offset + 3] * ws + g[g_offset + 4] * wt;
    ut[it] = g[g_offset + 2] * wr + g[g_offset + 4] * ws + g[g_offset + 5] * wt;
  }

  __syncthreads();

  for(int it = threadIdx.x; it < e_size; it += blockDim.x) {
    j = it / N;
    i = it - j * N;
    k = j / N;
    j -= k * N;
    double s = 0.0;
    for(int n = 0; n < N; ++n) {
      s += d_s[i * N + n] * ur[N * (j + N * k) + n] +
        d_s[j * N + n] * us[N * (n + N * k) + i] +
        d_s[k * N + n] * ut[N * (j + N * n) + i];
    }
    w[e_offset + it] = s;
  }
}

