#include <dlfcn.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <map>

static std::map<cudaStream_t, bool> sync;

typedef cudaError_t (*DEVICE_SYNC)(void);
typedef cudaError_t (*STREAM_SYNC)(cudaStream_t stream);
typedef cudaError_t (*MEMCPY_ASYNC)(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream);
typedef cudaError_t (*LAUNCH)(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream);

cudaError_t cudaDeviceSynchronize(void) {
  static void *handle = NULL;
  static DEVICE_SYNC cudaDeviceSynchronize = NULL;
  if (!handle) {
    handle = dlopen("libcudart.so", RTLD_NOW);
    cudaDeviceSynchronize = (DEVICE_SYNC)dlsym(handle, "cudaDeviceSynchronize");
  }
  bool all_sync = true;
  for (auto &iter : sync) {
    if (iter.second == false) {
      all_sync = false;
    }
  } 
  cudaError_t ret = cudaSuccess;
  if (all_sync == false) {
    ret = cudaDeviceSynchronize();
    for (auto &iter : sync) {
      iter.second = true;
    } 
  }
  return ret;
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
  static void *handle = NULL;
  static STREAM_SYNC real_cudaStreamSynchronize = NULL;
  if (!handle) {
    handle = dlopen("libcudart.so", RTLD_NOW);
    real_cudaStreamSynchronize = (STREAM_SYNC)dlsym(handle, "cudaStreamSynchronize");
  }
  cudaError_t ret = cudaSuccess;
  if (sync[stream] == false) {
    ret = real_cudaStreamSynchronize(stream);
    sync[stream] = true;
  }
  return ret;
}


cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream) {
  static void *handle = NULL;
  static LAUNCH real_cudaLaunchKernel = NULL;
  if (!handle) {
    handle = dlopen("libcudart.so", RTLD_NOW);
    real_cudaLaunchKernel = (LAUNCH)dlsym(handle, "cudaLaunchKernel");
  }
  cudaError_t ret = cudaSuccess;
  ret = real_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
  sync[stream] = false;
  return ret;
}


cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) {
  static void *handle = NULL;
  static MEMCPY_ASYNC real_cudaMemcpyAsync = NULL;
  if (!handle) {
    handle = dlopen("libcudart.so", RTLD_NOW);
    real_cudaMemcpyAsync = (MEMCPY_ASYNC)dlsym(handle, "cudaMemcpyAsync");
  }
  cudaError_t ret = cudaSuccess;
  ret = real_cudaMemcpyAsync(dst, src, count, kind, stream);
  sync[stream] = false;
  return ret;
}
