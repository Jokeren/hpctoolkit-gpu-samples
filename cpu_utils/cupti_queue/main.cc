#include <cstdio>
#include <cassert>
#include <omp.h>
#include "cupti_queue.h"

static int const NUM = 10;
static int const ITER = 10000;

int main() {
  CuptiQueue<int> q1, q2, q3;
  int order = 1;
  int total = NUM * ITER;
  #pragma omp parallel
  {
    if (omp_get_thread_num() == 0) {
      for (size_t j = 0; j < ITER; ++j) {
        for (size_t i = 0; i < NUM; ++i) {
          q1.insert(order);
          ++order;
        }
      }
    } else if (omp_get_thread_num() == 1) {
      while (total > 0) {
        q2.splice(q1);
        CuptiQueue<int>::item_type *num;
        for (size_t i = 0; i < NUM / 2; ++i) {
          if (num = q2.pop()) {
            assert(num->value() == NUM * ITER - total + 1);
            --total;
          } else {
            break;
          }
        }
      }
    } else {
      printf("Please use OMP_NUM_THREADS=2\n");
    }
  }
}
