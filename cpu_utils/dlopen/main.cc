#include <dlfcn.h>
#include <pthread.h>
#include <cstdio>

typedef int (*FIB)(int n);

int main() {
  static void *handle = NULL;
  static FIB fib = NULL;

  handle = dlopen("./fib.so", RTLD_NOW);
  fib = (FIB)dlsym(handle, "fib");
  printf("fib(20): %d\n", fib(20));

  return 0;
}
