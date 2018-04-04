#!/bin/bash

export OMP_NUM_THREADS=2

for((i=0;i<100;++i))
do
  echo "Iteration: "$i
  ./main
done
