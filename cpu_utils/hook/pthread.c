#include <stdio.h>
#include <string.h>
#include <dlfcn.h>
#include <pthread.h>

typedef void *pthread_start_fcn_t(void *);

typedef int (*PTHREAD_CREATE)(pthread_t *thread, const pthread_attr_t *attr, pthread_start_fcn_t *start_routine, void *arg);

static pthread_mutex_t mutex;

int pthread_create(pthread_t *thread, const pthread_attr_t *attr, pthread_start_fcn_t *start_routine,	void *arg) {
  static void *handle = NULL;
  static PTHREAD_CREATE old_pthread_create = NULL;
  printf("hook pthread_create\n");
  pthread_mutex_lock(&mutex);
  if (!handle) {
    handle = dlopen("libpthread.so.0", RTLD_NOW);
    old_pthread_create = (PTHREAD_CREATE)dlsym(handle, "pthread_create");
  }
  int ret = old_pthread_create(thread, attr, start_routine, arg);
  pthread_mutex_unlock(&mutex);
  return ret;
}

