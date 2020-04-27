#!/bin/bash

export HPCTOOLKIT_GPU_TEST_REP=10000

for spt in 16 #1 2 4 8 16
do
  for ((ctx = 2; ctx < 4; ctx++))
  do
    for ((stm = 10; stm <= 80; stm+=20))
    do
      export STREAMS_PER_THREAD=$spt
      export NUM_CONTEXTS=$ctx
      export NUM_STREAMS_PER_CONTEXT=$stm
      export OMP_NUM_THREADS=$(expr "$ctx" '*' "$stm")

      echo "-------------------------------------------------------"
      echo "STREAMS_PER_THREAD = "$STREAMS_PER_THREAD
      echo "HPCTOOLKIT_GPU_TEST_REP = "$HPCTOOLKIT_GPU_TEST_REP
      echo "NUM_CONTEXTS = "$NUM_CONTEXTS
      echo "NUM_STREAMS_PER_CONTEXT = "$NUM_STREAMS_PER_CONTEXT
      echo "OMP_NUM_THREADS = "$OMP_NUM_THREADS
      echo "-------------------------------------------------------"

      hpcrun -e gpu=nvidia -t ./main
    done
  done
done


