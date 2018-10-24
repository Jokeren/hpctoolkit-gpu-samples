__global__
void vecAdd(float *l, float *r, float *result, size_t N) {
  size_t i = threadIdx.x;
LABEL1:
  if (l[i] > i) {
    result[i] = l[i] - r[i];
  } else {
LABEL2:
    result[i] = l[i] + r[i];
  }
  if (i < 5) {
    ++i;
    goto LABEL2;
  } else if (i < 10) {
    ++i;
    goto LABEL1;
  }
}

