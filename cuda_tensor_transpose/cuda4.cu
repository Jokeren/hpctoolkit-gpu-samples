#include <cuda_runtime.h>
#include <cuda.h>

#define TILE_SIZE 5900
#define NTHREADS 512

__global__
void tensor_transpose(int dim_input, int dim_output, int nblocks, int tile_size,
  double *input, double *output) {
  __shared__ double tile[TILE_SIZE];

  for (int block_idx = blockIdx.x; block_idx < nblocks; block_idx += gridDim.x) {
    int it = block_idx, im = 0, offset1 = 0;
    for (int i = 0; i < dim_input; i++) {
      im = it * d_shape_input_r[i];
      offset1 += d_stride_input[i] * (it - im * d_shape_input[i]);
      it = im;
    }

    for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
      tile[i] = input[i + block_idx * tile_size];
    }

    __syncthreads();
  
    for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
      it = i;
      int offset2 = 0, local_offset = 0;
      for (int j = 0; j < dim_output; j++) {
        im = it * d_shape_output_r[j];
        int tmp = it - im * d_shape_output[j];
        offset2 += d_stride_output_global[j] * tmp;
        local_offset += d_stride_output_local[j] * tmp;
        it = im;
      }
      output[offset1 + offset2] = tile[local_offset];
    }

    __syncthreads();
  }
}
