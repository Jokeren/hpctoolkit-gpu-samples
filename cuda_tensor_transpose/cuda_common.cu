const int nblocks = d4 * d5 * d6;
const int tile_size = d1 * d2 * d3;
const int dim_output = 3;
const int dim_input = 3;
double *device_output, *device_input;

#if defined CUDA4 || defined CUDA5
cudaMalloc(&device_output, data_size * sizeof(double));
cudaMalloc(&device_input, data_size * sizeof(double));
cudaMemcpy(device_input, input, data_size * sizeof(double), cudaMemcpyHostToDevice );
cudaMemcpyToSymbol(d_shape_input, shape_input, dim_input * sizeof(int));
cudaMemcpyToSymbol(d_shape_input_r, shape_input_r, dim_input * sizeof(float));
cudaMemcpyToSymbol(d_shape_output, shape_output, dim_output * sizeof(int));
cudaMemcpyToSymbol(d_shape_output_r, shape_output_r, dim_output * sizeof(float));
cudaMemcpyToSymbol(d_stride_input, stride_input, dim_input * sizeof(int));
cudaMemcpyToSymbol(d_stride_output_local, stride_output_local, dim_output * sizeof(int));
cudaMemcpyToSymbol(d_stride_output_global, stride_output_global, dim_output * sizeof(int));

#else
int *device_shape_input, *device_shape_output;
float *device_shape_input_r, *device_shape_output_r;
int *device_stride_output_local, *device_stride_output_global;
int *device_stride_input;

cudaMalloc(&device_output, data_size * sizeof(double));
cudaMalloc(&device_input, data_size * sizeof(double));
cudaMalloc(&device_shape_input, dim_input * sizeof(int));
cudaMalloc(&device_shape_input_r, dim_input * sizeof(float));
cudaMalloc(&device_shape_output, dim_output * sizeof(int));
cudaMalloc(&device_shape_output_r, dim_output * sizeof(float));
cudaMalloc(&device_stride_input, dim_input * sizeof(int));
cudaMalloc(&device_stride_output_local, dim_output * sizeof(int));
cudaMalloc(&device_stride_output_global, dim_output * sizeof(int));

cudaMemcpy(device_input, input, data_size * sizeof(double), cudaMemcpyHostToDevice );
cudaMemcpy(device_shape_input, shape_input, dim_input * sizeof(int), cudaMemcpyHostToDevice );
cudaMemcpy(device_shape_input_r, shape_input_r, dim_input * sizeof(float), cudaMemcpyHostToDevice );
cudaMemcpy(device_shape_output, shape_output, dim_output * sizeof(int), cudaMemcpyHostToDevice );
cudaMemcpy(device_shape_output_r, shape_output_r, dim_output * sizeof(float), cudaMemcpyHostToDevice );
cudaMemcpy(device_stride_input, stride_input, dim_input * sizeof(int), cudaMemcpyHostToDevice );
cudaMemcpy(device_stride_output_local, stride_output_local, dim_output * sizeof(int), cudaMemcpyHostToDevice );
cudaMemcpy(device_stride_output_global, stride_output_global, dim_output * sizeof(int), cudaMemcpyHostToDevice );
#endif

cudaEvent_t event_start, event_end;
cudaEventCreate(&event_start);
cudaEventCreate(&event_end);

cudaEventRecord(event_start);
for (size_t i = 0; i < ITER; ++i) {
#if defined CUDA4
tensor_transpose<<<nblocks, NTHREADS>>>(dim_input, dim_output, nblocks, tile_size,
                                        device_input, device_output);
#elif defined CUDA5
tensor_transpose<dim_input, dim_output><<<nblocks, NTHREADS>>>(nblocks, tile_size,
                                        device_input, device_output);
#else
tensor_transpose<<<nblocks, NTHREADS>>>(dim_input, dim_output, nblocks, tile_size,
                                        device_shape_input, device_shape_output,
                                        device_shape_input_r, device_shape_output_r,
                                        device_stride_input, device_stride_output_local, device_stride_output_global,
                                        device_input, device_output);
#endif
}
cudaEventRecord(event_end);
cudaEventSynchronize(event_end);

cudaEventElapsedTime(&elapsed_time, event_start, event_end);
elapsed_time /= 1000.0;

cudaMemcpy(output, device_output, data_size * sizeof(double), cudaMemcpyDeviceToHost);

#if defined CUDA4 || defined CUDA5
cudaFree(device_output);
cudaFree(device_input);
#else
cudaFree(device_output);
cudaFree(device_input);
cudaFree(device_shape_input);
cudaFree(device_shape_input_r);
cudaFree(device_shape_output);
cudaFree(device_shape_output_r);
cudaFree(device_stride_input);
cudaFree(device_stride_output_local);
cudaFree(device_stride_output_global);
#endif
