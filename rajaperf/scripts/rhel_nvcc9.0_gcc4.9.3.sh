#!/bin/bash

##
## Copyright (c) 2017-18, Lawrence Livermore National Security, LLC.
##
## Produced at the Lawrence Livermore National Laboratory.
##
## LLNL-CODE-738930
##
## All rights reserved.
## 
## This file is part of the RAJA Performance Suite.
##
## For details about use and distribution, please read raja-perfsuite/LICENSE.
##

CUDA=/usr/local/cuda-9.0

rm -rf build_rhel_nvcc9.0_gcc4.9.3 >/dev/null
mkdir build_rhel_nvcc9.0_gcc4.9.3 && cd build_rhel_nvcc9.0_gcc4.9.3


  -DRAJA_HAVE_MM_MALLOC=0 \
  -DCMAKE_HAVE_PTHREADS_CREATE=0 \

cmake \
  -DCMAKE_BUILD_TYPE=Debug \
   -C ../host-configs/rhel/nvcc_gcc_4_9_3.cmake \
  -DENABLE_OPENMP=Off \
  -DCUB_INCLUDE_DIRS=/projects/pkgs-src/cub-1.8.0 \
  -DENABLE_CUDA=On \
  -DCUDA_TOOLKIT_ROOT_DIR=${CUDA} \
  -DPERFSUITE_ENABLE_WARNINGS=Off \
  -DENABLE_ALL_WARNINGS=Off \
  -DCMAKE_INSTALL_PREFIX=../install_rhel_nvcc9.0_gcc4.9.3 \
  "$@" \
  ..
