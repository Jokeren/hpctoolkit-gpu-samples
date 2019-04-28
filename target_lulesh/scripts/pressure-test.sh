#!/bin/bash

export OMP_NUM_THREADS=2

# Small workload & high sampling rate
for((i=0;i<5;i++))
do
  hpcrun -e cycles -e nvidia-ompt-pc-sampling@5 ./lulesh -i 2
  rm -rf hpctoolkit-*
done


# Small workload & low sampling rate
for((i=0;i<5;i++))
do
  hpcrun -e cycles -e nvidia-ompt-pc-sampling@10 ./lulesh -i 2
  rm -rf hpctoolkit-*
done


# Medium workload & high sampling rate
for((i=0;i<5;i++))
do
  hpcrun -e cycles -e nvidia-ompt-pc-sampling@5 ./lulesh -i 5
  rm -rf hpctoolkit-*
done


# Medium workload & low sampling rate
for((i=0;i<5;i++))
do
  hpcrun -e cycles -e nvidia-ompt-pc-sampling@10 ./lulesh -i 5
  rm -rf hpctoolkit-*
done


# Stand alone CPU
hpcrun -e cycles ./lulesh -i 5
rm -rf hpctoolkit-*


# Stand alone GPU1
hpcrun -e cycles -e nvidia-ompt-pc-sampling@10 ./lulesh -i 5
rm -rf hpctoolkit-*


# Stand alone GPU2
hpcrun -e cycles -e nvidia-ompt@10 ./lulesh -i 5
rm -rf hpctoolkit-*


# Full workload
hpcrun -e cycles -e nvidia-ompt-pc-sampling@10 ./lulesh -i 10
rm -rf hpctoolkit-*

# Full workload more threads
export OMP_NUM_THREADS=10
hpcrun -e cycles -e nvidia-ompt-pc-sampling@10 ./lulesh -i 10
rm -rf hpctoolkit-*
