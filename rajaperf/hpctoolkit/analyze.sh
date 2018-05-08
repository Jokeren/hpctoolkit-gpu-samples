#!/bin/bash

hpcstructdir=hpctoolkit-raja-perf-nolibs.exe.hpcstructs
hpcmeasurementdir=hpctoolkit-raja-perf-nolibs.exe-measurements
cubins=(`ls ${hpcmeasurementdir}/cubins`)
rm -rf ${hpcstructdir}
mkdir ${hpcstructdir}

#hpcstruct
for((i=0;i<${#cubins[@]};i++))
do
  cubin=${cubins[$i]}
  hpcstruct ${hpcmeasurementdir}/cubins/${cubin}
  mv ${cubin}.hpcstruct ${hpcstructdir}/${cubin}.hpcstruct
done

#hpcprof
structs=(`ls ${hpcstructdir}`)
hpcprofcommand="hpcprof"
for((i=0;i<${#structs[@]};i++))
do
  struct=${hpcstructdir}/${structs[$i]}
  hpcprofcommand=${hpcprofcommand}" -S "${struct}
done

hpcprofcommand=${hpcprofcommand}" "${hpcmeasurementdir}
`${hpcprofcommand}`
