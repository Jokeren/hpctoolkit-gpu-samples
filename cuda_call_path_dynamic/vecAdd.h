#ifndef _VEC_ADD_H_
#define _VEC_ADD_H_

__global__ void vecAdd(int *l, int *r, int *p, size_t i, size_t N);

__global__ void add(int *l, int *r, int *p, size_t i);

#endif
