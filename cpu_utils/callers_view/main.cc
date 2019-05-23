#include <cstdio>

static const size_t N = 1000000;
static int p1[N];
static int p2[N];

__attribute__ ((always_inline)) void add(int *a, int *b, size_t i) {
  a[i] = a[i] + b[i];
  b[i] = a[i] + b[i];
  a[i] = a[i] * b[i];
  b[i] = a[i] * b[i];
  a[i] = a[i] + b[i];
  b[i] = a[i] + b[i];
  a[i] = a[i] * b[i];
  b[i] = a[i] * b[i];
}


__attribute__ ((always_inline)) void trans_add(int *a, int *b, size_t i) {
  a[i] = a[i] + b[i];
  b[i] = a[i] + b[i];
  a[i] = a[i];
  b[i] = a[i];
  a[i] = a[i] + b[i];
  b[i] = a[i] + b[i];
  a[i] = a[i];
  b[i] = a[i];
  add(a, b, i);
}

__attribute__ ((noinline)) void noinline_add1(int *a, int *b, size_t i) {
  trans_add(a, b, i);
}

__attribute__ ((noinline)) void noinline_add2(int *a, int *b, size_t i) {
  trans_add(a, b, i);
}

int main() {
  for (size_t i = 0; i < N; ++i) {
    p1[i] = i;
    p2[i] = i + 1;
  }

  for (size_t i = 0; i < N; ++i) {
    noinline_add1(p1, p2, i);
  }

  for (size_t i = 0; i < N; ++i) {
    p1[i] = i - 1;
    p2[i] = i;
  }

  for (size_t i = 0; i < N; ++i) {
    noinline_add2(p1, p2, i);
  }

  printf("%d\n", p1[10]);

  return 0;
}
