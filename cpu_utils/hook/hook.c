#include <stdio.h>
#include <string.h>
#include <dlfcn.h>
#include <pthread.h>

typedef int (*SETENV)(const char *name, const char *value, int overwrite);
typedef char *(*GETENV)(const char *name);

static pthread_mutex_t mutex;

int setenv(const char *name, const char *value, int overwrite) {
  static void *handle = NULL;
  static SETENV old_setenv = NULL;
  pthread_mutex_lock(&mutex);
  if (!handle) {
    handle = dlopen("libc.so.6", RTLD_NOW);
    old_setenv = (SETENV)dlsym(handle, "setenv");
  }
  int ret = old_setenv(name, value, overwrite);
  pthread_mutex_unlock(&mutex);
  return ret;
}


char *getenv(const char *name) {
  static void *handle = NULL;
  static GETENV old_getenv = NULL;
  pthread_mutex_lock(&mutex);
  if (!handle) {
    handle = dlopen("libc.so.6", RTLD_NOW);
    old_getenv = (GETENV)dlsym(handle, "getenv");
  }
  char *ret = old_getenv(name);
  pthread_mutex_unlock(&mutex);
  return ret;
}
