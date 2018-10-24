#!/bin/bash

echo "Configure cuda-10.0"
make clean
make ARCH=sm_60 CUDA_DIR=/usr/local/cuda-10.0
/usr/local/cuda-10.0/bin/nvdisasm -poff -cfg vecAdd.cubin &> sm_60_cuda_10.0.dot

echo "Configure cuda-9.2"
make clean
make ARCH=sm_60 CUDA_DIR=/usr/local/cuda-9.2
/usr/local/cuda-9.2/bin/nvdisasm -poff -cfg vecAdd.cubin &> sm_60_cuda_9.2.dot
