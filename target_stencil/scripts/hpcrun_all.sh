#!/bin/bash

for((i=3;i<6;i++))
do
  rm -rf hpctoolkit-main-measurements
  ./scripts/hpcrun.sh $i
  hpcstruct hpctoolkit-main-measurements
  hpcstruct main
  hpcprof -S main.hpcstruct hpctoolkit-main-measurements
  mv hpctoolkit-main-database hpctoolkit-main-database-$i
  mv hpctoolkit-main-measurements hpctoolkit-main-measurements-$i
done
