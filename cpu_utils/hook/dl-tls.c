#include <stdio.h>
#include <dlfcn.h>
#include <pthread.h>

typedef void *(*DL_ALLOCATE_TLS)(void *mem);
typedef void *(*DL_ALLOCATE_TLS_INIT)(void *result);
typedef void *(*DL_ALLOCATE_TLS_STORAGE)(void);

static pthread_mutex_t mutex;

void *
_dl_allocate_tls(void *mem) {
  static void *handle = NULL;
  static DL_ALLOCATE_TLS old_dl_allocate_tls = NULL;
  printf("hook _dl_allocate_tls\n");
  pthread_mutex_lock(&mutex);
  if (!handle) {
    handle = dlopen("libc.so.6", RTLD_NOW);
    old_dl_allocate_tls = (DL_ALLOCATE_TLS)dlsym(handle, "_dl_allocate_tls");
  }
  void *ret = old_dl_allocate_tls(mem);
  pthread_mutex_unlock(&mutex);
  return ret;
}


void *
_dl_allocate_tls_init(void *mem) {
  static void *handle = NULL;
  static DL_ALLOCATE_TLS_INIT old_dl_allocate_tls_init = NULL;
  printf("hook _dl_allocate_tls_init\n");
  pthread_mutex_lock(&mutex);
  if (!handle) {
    handle = dlopen("libc.so.6", RTLD_NOW);
    old_dl_allocate_tls_init = (DL_ALLOCATE_TLS_INIT)dlsym(handle, "_dl_allocate_tls_init");
  }
  void *ret = old_dl_allocate_tls_init(mem);
  pthread_mutex_unlock(&mutex);
  return ret;
}


void *
_dl_allocate_tls_storage(void) {
  static void *handle = NULL;
  static DL_ALLOCATE_TLS_STORAGE old_dl_allocate_tls_storage = NULL;
  printf("hook _dl_allocate_tls_storage\n");
  pthread_mutex_lock(&mutex);
  if (!handle) {
    handle = dlopen("libc.so.6", RTLD_NOW);
    old_dl_allocate_tls_storage = (DL_ALLOCATE_TLS_STORAGE)dlsym(handle, "_dl_allocate_tls_storage");
  }
  void *ret = old_dl_allocate_tls_storage();
  pthread_mutex_unlock(&mutex);
  return ret;
}
