#!/bin/bash

N=$1
make clean
make SHOWFLAG="-DOPT"$N
echo "OPT"$N
./main
