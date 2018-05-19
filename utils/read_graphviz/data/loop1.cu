__global__
void vecAdd(float *l, float *r, float *result, size_t N) {
  size_t i = threadIdx.x;
  result[i] = l[i] + r[i];
}
