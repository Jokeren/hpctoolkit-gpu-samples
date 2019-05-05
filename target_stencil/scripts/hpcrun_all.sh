#!/bin/bash

for((i=0;i<7;i++))
do
  rm -rf hpctoolkit-main-measurements
  ./scripts/hpcrun.sh $i
  hpcstruct hpctoolkit-main-measurements
  hpcstruct main
  hpcprof -S main.hpcstruct hpctoolkit-main-measurements
  mv hpctoolkit-main-database hpctoolkit-main-database-$i
done
