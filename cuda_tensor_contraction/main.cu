#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <omp.h>

static const int E = 50000;
static const int N = 10;
static const int B = 512;

// Deprecated GPU initialization
//__global__
//void k_init_input(double *u, double *gxyz, int N) {
//  int ielt = blockIdx.x;
//  int sz_elt = N * N * N;
//  int elt_offset = sz_elt*ielt;
//
//  for(int it = threadIdx.x ; it < sz_elt ; it += blockDim.x)
//  {
//    int j=it/N; int i=it-j*N; int k=j/N; j-=k*N;
//    u[it+elt_offset] = 0.1*((i+1)+(j+1)+(k+1))+(ielt+1)*100;
//    for(int p=0;p<6;p++)
//      gxyz[p+6*(it+elt_offset)]=(p+1)+(i+1)+(j+1)+(k+1)+(ielt+1)*1000;
//  }
//}

// Original CUDA kernel
//__global__ void k_nekbone_ax(double *w, double *u, double *gxyz, 
//  double *dxm1, double *dxtm1, int np)
//{
//  int sz_elt=np*np*np;
//  int elt_offset=sz_elt*blockIdx.x; // directly map block id to element
//  __shared__ double ur[1024];
//  __shared__ double us[1024];
//  __shared__ double ut[1024];
//  __shared__ double ul[1024]; // 25% speedup
//  int it,i,j,k,p,gi0;
//  double s,wr,ws,wt;
//
//  for(it=threadIdx.x ; it<sz_elt ; it+=blockDim.x)
//    ul[it]=u[it+elt_offset];
//
//  __syncthreads();
//
//  for(it=threadIdx.x ; it<sz_elt ; it+=blockDim.x)
//  {
//    j=it/np; i=it-j*np; k=j/np; j-=k*np;
//    wr=0.; ws=0.; wt=0.;
//    for(p=0;p<np;p++)
//    {
//      wr+=dxtm1[p+i*np]*ul[p+np*(j+np*k)];
//      ws+=dxtm1[p+j*np]*ul[i+np*(p+np*k)];
//      wt+=dxtm1[p+k*np]*ul[i+np*(j+np*p)];
//    }
//    gi0=6*(it+elt_offset);
//    ur[it] = gxyz[0+gi0]*wr + gxyz[1+gi0]*ws + gxyz[2+gi0]*wt;
//    us[it] = gxyz[1+gi0]*wr + gxyz[3+gi0]*ws + gxyz[4+gi0]*wt;
//    ut[it] = gxyz[2+gi0]*wr + gxyz[4+gi0]*ws + gxyz[5+gi0]*wt;
//  }
//
//  __syncthreads();
//
//  for(it=threadIdx.x ; it<sz_elt ; it+=blockDim.x)
//  {
//    j=it/np; i=it-j*np; k=j/np; j-=k*np;
//    s=0.;
//    for(p=0;p<np;p++)
//      s+=dxm1[p+i*np]*ur[p+np*(j+np*k)]
//        +dxm1[p+j*np]*us[i+np*(p+np*k)]
//        +dxm1[p+k*np]*ut[i+np*(j+np*p)];
//    w[it+elt_offset] = s;
//  }
//}

void init(double *u, double *g, double *d, double *dt) {
  for(int j = 0; j < N; j++) {
    for(int i = 0; i < N; i++) {
      dt[i * N + j] = d[j * N + i] = (i + 1) * (i + 1) + (j + 1);
    }
  }
  
  #pragma omp parallel for
  for (size_t e = 0; e < E; ++e) {
    size_t e_offset = e * N * N * N;
    for (size_t k = 0; k < N; ++k) {
      for (size_t j = 0; j < N; ++j) {
        for (size_t i = 0; i < N; ++i) {
          size_t offset = k * N * N + j * N + i;
          u[e_offset + offset] = 0.1 * ((i + 1) + (j + 1) + (k + 1)) + (e + 1) * 100;
          for (size_t p = 0; p < 6; ++p) {
            g[6 * (e_offset + offset) + p] = (p + 1) + (i + 1) + (j + 1) + (k + 1) + (e + 1) * 1000;
          }
        }
      }
    }
  }
}


__global__
void nekbone(double *w, double *u, double *g, double *d, double *dt) {
  const int e_size = N * N * N;
  const int e_offset = e_size * blockIdx.x;

  __shared__ double ur[1024];
  __shared__ double us[1024];
  __shared__ double ut[1024];
  __shared__ double ul[1024];

  for (int it = threadIdx.x ; it < e_size; it += blockDim.x) {
    ul[it] = u[e_offset + it];
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

  for(int it = threadIdx.x; it < e_size; it += blockDim.x) {
    j = it / N;
    i = it - j * N;
    k = j / N;
    j -= k * N;
    double s = 0.0;
    for(int n = 0; n < N; ++n) {
      s += d[i * N + n] * ur[N * (j + N * k) + n] +
        d[j * N + n] * us[N * (n + N * k) + i] +
        d[k * N + n] * ut[N * (j + N * n) + i];
    }
    w[e_offset + it] = s;
  }
}


int main() {
  double *w, *d, *dt, *g, *u;
  double *w_d, *d_d, *dt_d, *g_d, *u_d;

  g  = (double *)calloc(6 * N * N * N * E, sizeof(double));
  u  = (double *)calloc(N * N * N * E, sizeof(double));
  w  = (double *)calloc(N * N * N * E, sizeof(double));
  d  = (double *)calloc(N * N, sizeof(double));
  dt = (double *)calloc(N * N, sizeof(double));

  cudaMalloc<double>(&g_d, sizeof(double) * 6 * N * N * N * E);
  cudaMalloc<double>(&u_d, sizeof(double) * N * N * N * E);
  cudaMalloc<double>(&w_d, sizeof(double) * N * N * N * E);
  cudaMalloc<double>(&d_d, sizeof(double) * N * N);
  cudaMalloc<double>(&dt_d, sizeof(double) * N * N);
  
  init(u, g, d, dt);

  cudaMemcpy(g_d, g, sizeof(double) * 6 * N * N * N * E, cudaMemcpyHostToDevice);
  cudaMemcpy(u_d, u, sizeof(double) * N * N * N * E, cudaMemcpyHostToDevice);
  cudaMemcpy(d_d, d, sizeof(double) * N * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dt_d, dt, sizeof(double) * N * N, cudaMemcpyHostToDevice);

  float elapsed_time = 0.0;
  cudaEvent_t event_start, event_end;
  cudaEventCreate(&event_start);
  cudaEventCreate(&event_end);

  cudaEventRecord(event_start);

  nekbone<<<E, B>>>(w_d, u_d, g_d, d_d, dt_d);

  cudaEventRecord(event_end);
  cudaEventSynchronize(event_end);

  cudaEventElapsedTime(&elapsed_time, event_start, event_end);
  elapsed_time /= 1000.0;

  printf("kernel time (s): %f\n", elapsed_time);

  cudaMemcpy(w, w_d, sizeof(double) * N * N * N * E, cudaMemcpyDeviceToHost);

  printf("First 5 sums:\n");
  //2.143933e+14  8.402399e+14  1.877629e+15  3.326562e+15  5.187038e+15  
  
  int it = 0;
  int it_next = 0;
  for(size_t i = 0; i < 5; ++i) {
    it_next += N * N * N;
    double s = 0.0;
    for(; it < it_next; it++) {
      s += w[it];
    }
    printf("%14.6e",s);
  }
  printf("\n");

  free(g);
  free(u);
  free(w);
  free(d);
  free(dt);

  cudaFree(g_d);
  cudaFree(u_d);
  cudaFree(w_d);
  cudaFree(d_d);
  cudaFree(dt_d);

  return 0;
}













