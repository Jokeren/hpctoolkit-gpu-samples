// A proxy app for testing the correctness of parsing `call` notation
#include <cstdio>

static const int N = 100000;
static int A[100];
static int ans = 100;

void __attribute__ ((noinline)) func()
{
  for (int i = 0; i < 100; ++i) {
    A[i] = (ans / 2);
    ans = ans / 2;
  }
}

int main(int argc, char **argv)
{
  for (int i = 0; i < N; ++i) {
    func();
  }
  printf("ans %d\n", ans);
  return 0;
}

