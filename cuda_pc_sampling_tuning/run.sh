#!/bin/bash

CUPTI_DEVICE_NUM=1
CUPTI_SAMPLING_PERIOD=1
SCRIPT=../cupti_test/cupti-preload/timesampling
BLOCKS=$1
THREADS=$2

CUPTI_DEVICE_NUM=$CUPTI_DEVICE_NUM CUPTI_SAMPLING_PERIOD=$CUPTI_SAMPLING_PERIOD $SCRIPT ./main $CUPTI_DEVICE_NUM $BLOCKS $THREADS
