#include <cstdio>
#include <cassert>
#include <thread>
#include <chrono>
#include <omp.h>
#include "cupti_stack.h"
#include "cupti_record.h"

void stack_test() {
  const int ITER = 1000;
  const int NUM = 1000;

  CuptiStack<int> s1, s2, s3;
  int order = 1;
  int total = NUM * ITER;
  #pragma omp parallel
  {
    printf("[Stack Test]: Thread %d\n", omp_get_thread_num());
    if (omp_get_thread_num() == 0) {
      for (size_t j = 0; j < ITER; ++j) {
        for (size_t i = 0; i < NUM; ++i) {
          CuptiStackItem<int> *item = new CuptiStackItem<int>(order);
          s1.push(item);
          ++order;
        }
        //std::this_thread::sleep_for(std::chrono::milliseconds(10));
        //std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    } else if (omp_get_thread_num() == 1) {
      while (s3.size() != total) {
        s2.set_top(s1.steal());
        s3.splice(s2);
      }
    } else {
      printf("[Stack Test]: Please use OMP_NUM_THREADS=2\n");
      exit(1);
    }
  }

  for (auto iter = s3.begin(); iter != s3.end(); ++iter) {
    int curr = *iter;
    if (curr != total) {
      printf("[Stack Test]: Error! %d<!=>%d\n", total, curr);
      break;
    }
    --total;
  }
}


void record_test() {
  const int ITER = 1000;
  const int NUM = 1000;

  CuptiRecord<int> cupti_record;
  int order = 1;
  int total = NUM * ITER;
  #pragma omp parallel
  {
    printf("[Record Test]: Thread %d\n", omp_get_thread_num());
    if (omp_get_thread_num() == 0) {
      for (size_t j = 0; j < ITER; ++j) {
        for (size_t i = 0; i < NUM; ++i) {
          cupti_record.worker_notification_apply(order);
          ++order;
        }
        //std::this_thread::sleep_for(std::chrono::milliseconds(10));
        //std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    } else if (omp_get_thread_num() == 1) {
      while (total > 0) {
        size_t size = cupti_record.cupti_notification_apply(cupti_record);
        total -= size;
      }
    } else {
      printf("[Record Test]: Please use OMP_NUM_THREADS=2\n");
      exit(1);
    }
  }

  printf("[Stack Test]: Recycled nodes %d\n", cupti_record.recycled);
}


int main() {
  stack_test();
  record_test();
}
