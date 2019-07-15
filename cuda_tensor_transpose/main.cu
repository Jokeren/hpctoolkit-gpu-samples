#include <cstdio>
#include <omp.h>

// 1,2,3,4,5,6 -> 2,3,4,6,1,5
static const int d1 = 41, d2 = 13, d3 = 11, d4 = 9, d5 = 76, d6 = 50;
static const int data_size = d1 * d2 * d3 * d4 * d5 * d6;
static int ITER = 1;

#if defined CUDA4 || defined CUDA5
__constant__ int d_shape_output[3];
__constant__ int d_shape_input[3];
__constant__ float d_shape_output_r[3];
__constant__ float d_shape_input_r[3];
__constant__ int d_stride_output_local[3];
__constant__ int d_stride_output_global[3];
__constant__ int d_stride_input[3];
#endif

static const int shape_output[] = {d2, d3, d1};
static const int shape_input[] = {d4, d5, d6};
static const float shape_output_r[] = {1.0 / d2, 1.0 / d3, 1.0 / d1};
static const float shape_input_r[] = {1.0 / d4, 1.0 / d5, 1.0 / d6};
static const int stride_output_local[] = {d1, d1 * d2, 1};
static const int stride_output_global[] = {1, d2, d2 * d3 * d4 * d6};
static const int stride_input[] = {d2 * d3, d2 * d3 * d4 * d6 * d1, d2 * d3 * d4};

void verify(double *input, double *output) {
  int input_offset  = 2 + d1 * (2 + d2 * (2 + d3 * (2 + d4 * (0 + 2 * d5))));
  int output_offset = 2 + d2 * (2 + d3 * (2 + d4 * (2 + d6 * (2 + 0 * d1))));
  for (size_t i = 0; i < d5; i++) {
    if (input[input_offset + i * d1 * d2 * d3 * d4] != output[output_offset + i * d2 * d3 * d4 * d6 * d1]) {
      printf("Failed!\n");
      exit(-1);
    }
  }
}

#if defined CUDA1
#include "cuda1.cu"
#elif defined CUDA2
#include "cuda2.cu"
#elif defined CUDA3
#include "cuda3.cu"
#elif defined CUDA4
#include "cuda4.cu"
#elif defined CUDA5
#include "cuda5.cu"
#endif

int main(int argv, char **argc) {
  if (argv > 1) {
    ITER = atoi(argc[1]);
  }

  double *input = new double[data_size]();
  double *output = new double[data_size]();

  for (size_t i = 0; i < data_size; i++) {
    input[i] = i;
  }

  float elapsed_time = 0.0f;

#if defined CUDA1
#include "cuda_common.cu"
#elif defined CUDA2
#include "cuda_common.cu"
#elif defined CUDA3
#include "cuda_common.cu"
#elif defined CUDA4
#include "cuda_common.cu"
#elif defined CUDA5
#include "cuda_common.cu"
#endif

  verify(input, output);

  printf("Elapsed time %lf\n", elapsed_time); 

  delete [] input;
  delete [] output;

  return 0;
}
