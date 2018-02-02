#include <cstdio>

#include <omp.h>

static const size_t N = 1000;

void init(int *p, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    p[i] = i;
  }
}

void output(int *p, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    printf("index %zu: %d\n", i, p[i]);
  }
}

int main() {
  int l1[N], l2[N];
  int r1[N], r2[N];
  int p1[N], p2[N];
  init(l1, N);
  init(r1, N);
  init(l2, N);
  init(r2, N);
  #pragma omp parallel
  {
    if (omp_get_thread_num() == 0) {
      #pragma omp target
      #pragma omp teams num_teams(4) thread_limit(64)
      {
        #pragma omp distribute parallel for
        for (size_t i = 0; i < N; ++i) {
          p1[i] = l1[i] + r1[i];
        }
      }
    } else if (omp_get_thread_num() == 1) {
      #pragma omp target
      #pragma omp teams distribute parallel for num_teams(4) thread_limit(64)
      for (size_t i = 0; i < N; ++i) {
        p2[i] = l2[i] + r2[i];
      }
    }
  }
  output(p1, N);
  output(p2, N);
  return 0;
}
