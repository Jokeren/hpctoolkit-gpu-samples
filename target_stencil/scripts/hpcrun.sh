#!/bin/bash

N=$1
make clean
make SHOWFLAGS="-DOPT"$N" -g"
echo "OPT"$N
hpcrun -e nvidia-cuda-pc-sampling -e CPUTIME ./main
