#!/bin/bash

N=$1
make clean
make SHOWFLAGS="-DOPT"$N
echo "OPT"$N
./main
