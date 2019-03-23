#!/bin/bash
# Begin LSF directives
#BSUB -P CSC322
#BSUB -J laghos
#BSUB -o laghos.log%J
#BSUB -W 1:00
#BSUB -nnodes 4
# End LSF directives and begin shell commands

date
cp ./Laghos/cuda/laghos $MEMBERWORK/csc322
cp ./Laghos/data/square01_quad.mesh $MEMBERWORK/csc322
cd $MEMBERWORK/csc322
jsrun -n 1 -a 1 -g 1 hpcrun -e nvidia-cuda -e cycles ./laghos -p 0 -m ./square01_quad.mesh -rs 1 -tf 0.75 -pa
