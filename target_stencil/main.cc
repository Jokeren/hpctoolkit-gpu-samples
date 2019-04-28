#include <cstdio>
#include <cstring>
#include <cmath>
#include <sys/time.h>
#include <algorithm>
#include <limits>
#include <random>
#include <omp.h>
#include <cuda_runtime.h>

static const size_t N = 10000;
static const size_t ITER = 100;
static const float T = 0.2;

void init(float *p, size_t W, size_t H) {
  static std::default_random_engine e;
  static std::uniform_real_distribution<> dis(0, 1); // rage 0 - 1

  for (size_t h = 0; h < H; ++h) {
    for (size_t w = 0; w < W; ++w) {
      p[h * W + w] = dis(e);
    }
  }
}

void output(float *p, size_t W, size_t H) {
  for (size_t h = 0; h < H; ++h) {
    for (size_t w = 0; w < W; ++w) {
      printf("%f ", p[h * W + w]);
    }
    printf("\n");
  }
}


int main(int argc, char *argv[]) {
  const size_t H = N, W = N;
  float *A = new float[H * W];
  float *A_new = new float[H * W];

  printf("Available devices %d\n", omp_get_num_devices());

  init(A, H, W);

#ifdef DEBUG
  float *A_orig = new float[H * W];
  std::copy(A, A + H * W, A_orig);
#endif

  float error = std::numeric_limits<float>::max();
  size_t iter = 0;

  float elapsed_time = 0.0;
  timeval t1, t2;
  gettimeofday(&t1, NULL);

#if defined OPT0
#include "opt0.cc"
#elif defined OPT1
#include "opt1.cc"
#elif defined OPT2
#include "opt2.cc"
#elif defined OPT3
#include "opt3.cc"
#elif defined OPT4
#include "opt4.cc"
#elif defined OPT5
#include "opt5.cc"
#elif defined OPT6
#include "opt6.cc"
#endif

  gettimeofday(&t2, NULL);
  elapsed_time = (t2.tv_sec - t1.tv_sec) * 1000.0;
  elapsed_time += (t2.tv_usec - t1.tv_usec) / 1000.0;
  elapsed_time /= 1000.0;

  printf("Time %f\n", elapsed_time);

#ifdef DEBUG
  printf("A\n");
  output(A_orig, H, W);
  printf("A_new\n");
  output(A_new, H, W);
#endif

  delete [] A;
  delete [] A_new;

  return 0;
}
