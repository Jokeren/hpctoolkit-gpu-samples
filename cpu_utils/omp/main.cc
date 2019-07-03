#include <cstdio>
#include <pthread.h>

static const int N = 10000000;
static const int TH = 4;
static int A[100];

void *func(void *id) {
  int tid = *((int *)id);
  printf("thread %d start\n", tid);
  for (int i = tid; i < N; i += TH) {
    A[i % 100] += i % 100 + 1;
  }
  return NULL;
}

int main(int argc, char **argv)
{
  pthread_t threads[TH];
  int ids[TH];
  for (size_t i = 0; i < TH; ++i) {
    ids[i] = i;
    pthread_create(threads + i, NULL, func, ids + i);
  }
  printf("all threads created\n");
  for (size_t i = 0; i < TH; ++i) {
    pthread_join(threads[i], NULL);
  }
  printf("ans %d\n", A[0]);
  return 0;
}
