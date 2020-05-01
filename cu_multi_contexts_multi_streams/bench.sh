#!/bin/bash

export HPCTOOLKIT_GPU_TEST_REP=10000
valgrind="/home/dx4/hpctools/spack/opt/spack/linux-rhel7-power9le/gcc-7.3.0/valgrind-3.15.0-kkdcxjgtl3gcc5l6lzdvbznj46dthaku/bin/valgrind"
subscribe="/home/dx4/cupti_test/cupti-preload/subscriber/subscribe"
enablesampling="/home/dx4/cupti_test/cupti-preload/pc_sampling/enablesampling"
hpcrun="../../hpctools/hpctoolkit/hpctoolkit-install/bin/hpcrun"


if_else=${1:-0}

if [ $if_else -eq 0 ]; then
	for spt in 1 2 4 16 32 64 128; do

		for ((ctx = 1; ctx <= 3; ctx++)); do

			for ((stm = 20; stm <= 80; stm += 20)); do

				export STREAMS_PER_THREAD=$spt
				export NUM_CONTEXTS=$ctx
				export NUM_STREAMS_PER_CONTEXT=$stm
				export OMP_NUM_THREADS=$(expr "$ctx" '*' "$stm")

				echo "-------------------------------------------------------"
				echo "STREAMS_PER_THREAD = "$STREAMS_PER_THREAD
				echo "-------------------------------------------------------"
				echo "HPCTOOLKIT_GPU_TEST_REP = "$HPCTOOLKIT_GPU_TEST_REP
				echo "NUM_CONTEXTS = "$NUM_CONTEXTS" ----------------------------------"
				echo "NUM_STREAMS_PER_CONTEXT = "$NUM_STREAMS_PER_CONTEXT
				echo "OMP_NUM_THREADS = "$OMP_NUM_THREADS

#				time enablesampling ./main
#				time subscribe ./main
				time hpcrun -e gpu=nvidia -t ./main -md
#				time ./main

			done
		done
	done

else

	export HPCTOOLKIT_GPU_TEST_REP=10000
	export STREAMS_PER_THREAD=16
	export NUM_CONTEXTS=3
	export NUM_STREAMS_PER_CONTEXT=60
	export OMP_NUM_THREADS=$(expr "$NUM_CONTEXTS" '*' "$NUM_STREAMS_PER_CONTEXT")

	echo "-------------------------------------------------------"
	echo "spt = "$STREAMS_PER_THREAD
	echo "-------------------------------------------------------"
	echo "HPCTOOLKIT_GPU_TEST_REP = "$HPCTOOLKIT_GPU_TEST_REP
	echo "NUM_CONTEXTS = "$NUM_CONTEXTS" ----------------------------------"
	echo "NUM_STREAMS_PER_CONTEXT = "$NUM_STREAMS_PER_CONTEXT
	echo "OMP_NUM_THREADS = "$OMP_NUM_THREADS

#	time enablesampling ./main
#	time subscribe ./main
#	time hpcrun -e gpu=nvidia -t ./main -md
	time ./main

fi
