#!/bin/bash

N=$1
make clean
make SHOWFLAG="-DOPT"$N
echo "OPT"$N
hpcrun -e nvidia-ompt-pc-sampling -e REALTIME ./main