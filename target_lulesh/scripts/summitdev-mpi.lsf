#!/bin/bash
# Begin LSF directives
#BSUB -P CSC322
#BSUB -J lulesh
#BSUB -o lulesh.log%J
#BSUB -W 1:00
#BSUB -nnodes 2
# End LSF directives and begin shell commands

date
jsrun -n 8 -a 1 -g 1 hpcrun -e nvidia-ompt ./lulesh2.0 -i 10
