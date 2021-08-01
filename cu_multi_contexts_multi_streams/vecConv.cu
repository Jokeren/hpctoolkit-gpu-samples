extern "C"
__global__
void vecConv(int *l, int *r, int *p, size_t N) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  size_t sum = 0;

  if ( idx < N ) {
    for(size_t i = 0; i <= idx; i++){
       sum += l[i] + r[idx-i];
    }
    p[idx] = sum;
  }
}
