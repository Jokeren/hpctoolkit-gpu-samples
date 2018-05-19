__global__
void vecAdd(float *l, float *r, float *result, size_t N) {
  size_t i = threadIdx.x;
LABEL:
  if (l[i] > i) {
    result[i] = exp(l[i]);
  } else {
    result[i] = acosf(l[i]);
  }
  if (i < 5) {
    ++i;
    l[i] = r[i] / 2.0;
    r[i] = r[i] / 2.0;
    goto LABEL;
  }
}

