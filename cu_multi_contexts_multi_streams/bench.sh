#!/bin/bash

usage()
{
    cat <<EOF
Usage:
    ./bench.sh [options] ./main
    options:
    -h help
    -c <num of contexts>
      default 1
    -d <cuda device>
      default 0 
    -n <number of elements>
      default 100000
    -r <recursive depth>
      default 0 
    -s <num of streams>
      default 1
    -t <num of threads>
      default 1
    -p <profiler command>
      default ""
    -dd <debug>
EOF
    exit 0
}

while test "x$1" != x
do
  arg="$1" ; shift
  case "$arg" in
    -c)
      export BENCH_NUM_CONTEXTS=$1
      shift
      ;;
    -d)
      export BENCH_DEVICE_ID=$1
      shift
      ;;
    -n)
      export BENCH_NUM_ELEMENTS=$1
      shift
      ;;
    -r)
      export BENCH_DEPTH=$1
      shift
      ;;
    -s)
      export BENCH_NUM_STREAMS_PER_CONTEXT=$1
      shift
      ;;
    -t)
      export BENCH_NUM_THREADS=$1
      shift
      ;;
    -p)
      export BENCH_PROFILER=$1
      shift
      ;;
    -dd)
      export BENCH_DEBUG=1
      ;;
    -h)
      usage
      exit
      ;;
    * )
      set -- "$arg" "$@"
      break
      ;;
  esac
done

echo $BENCH_PROFILER
time $BENCH_PROFILER ./main
