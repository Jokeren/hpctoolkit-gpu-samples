#!/bin/bash

# Small workload & high sampling rate
for((i=0;i<5;i++))
do
  mpirun -np 1 hpcrun -e cycles -e nvidia-cuda-pc-sampling@5 ./Laghos/cuda/laghos -p 0 -m ./Laghos/data/square01_quad.mesh --sync -rs 1 -tf 0.1 -pa
  rm -rf hpctoolkit-*
done


# Small workload & low sampling rate
for((i=0;i<5;i++))
do
  mpirun -np 1 hpcrun -e cycles -e nvidia-cuda-pc-sampling@10 ./Laghos/cuda/laghos -p 0 -m ./Laghos/data/square01_quad.mesh --sync -rs 1 -tf 0.1 -pa
  rm -rf hpctoolkit-*
done


# Medium workload & high sampling rate
for((i=0;i<5;i++))
do
  mpirun -np 1 hpcrun -e cycles -e nvidia-cuda-pc-sampling@5 ./Laghos/cuda/laghos -p 0 -m ./Laghos/data/square01_quad.mesh --sync -rs 1 -tf 0.75 -pa
  rm -rf hpctoolkit-*
done


# Medium workload & low sampling rate
for((i=0;i<5;i++))
do
  mpirun -np 1 hpcrun -e cycles -e nvidia-cuda-pc-sampling@10 ./Laghos/cuda/laghos -p 0 -m ./Laghos/data/square01_quad.mesh --sync -rs 1 -tf 0.75 -pa
  rm -rf hpctoolkit-*
done


# Stand alone CPU
mpirun -np 1 hpcrun -e cycles ./Laghos/cuda/laghos -p 0 -m ./Laghos/data/square01_quad.mesh --sync -rs 1 -tf 0.75 -pa
rm -rf hpctoolkit-*


# Stand alone GPU1
mpirun -np 1 hpcrun -e nvidia-cuda-pcsampling@10 ./Laghos/cuda/laghos -p 0 -m ./Laghos/data/square01_quad.mesh --sync -rs 1 -tf 0.75 -pa
rm -rf hpctoolkit-*


# Stand alone GPU2
mpirun -np 1 hpcrun -e nvidia-cuda ./Laghos/cuda/laghos -p 0 -m ./Laghos/data/square01_quad.mesh --sync -rs 1 -tf 0.75 -pa
rm -rf hpctoolkit-*


# Full workload
mpirun -np 1 hpcrun -e cycles -e nvidia-cuda-pc-sampling@10 ./Laghos/cuda/laghos -p 0 -m ./Laghos/data/square01_quad.mesh --sync -rs 3 -tf 0.75 -pa
rm -rf hpctoolkit-*
