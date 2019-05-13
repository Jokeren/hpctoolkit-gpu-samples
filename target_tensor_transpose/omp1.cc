#define TILE_SIZE 5900

const int nteams = d4 * d5 * d6;
const int tile_size = d1 * d2 * d3;
const int dim_output = 3;
const int dim_input = 3;

#pragma omp target data map(from:output[:data_size]) map(to:input[:data_size], \
    shape_input[:3], shape_output[:3], stride_output_local[:3], stride_output_global[:3], stride_input[:3])
{
  double t0 = omp_get_wtime();
  #pragma omp target teams distribute thread_limit(256)
  for (int team = 0; team < nteams; team++) {
    double tile[TILE_SIZE];

    int it = team, im = 0, offset1 = 0;
    for (int i = 0; i < dim_input; i++) {
      im = it / shape_input[i];
      offset1 += stride_input[i] * (it - im * shape_input[i]);
      it = im;
    }

    #pragma omp parallel for
    for (int i = 0; i < tile_size; i++) {
      tile[i] = input[i + team * tile_size];
    }

    #pragma omp parallel for
    for (int i = 0; i < tile_size; i++) {
      it = i;
      int offset2 = 0, local_offset = 0;
      for (int j = 0; j < dim_output; j++) {
        im = it / shape_output[j];
        int tmp = it - im * shape_output[j];
        offset2 += stride_output_global[j] * tmp;
        local_offset += stride_output_local[j] * tmp;
        it = im;
      }
      output[offset1 + offset2] = tile[local_offset];
    }
  }
  double t1 = omp_get_wtime();
  elapsed_time = t1 - t0;
}
