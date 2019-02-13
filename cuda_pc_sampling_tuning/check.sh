#!/bin/bash

rm -rf logs
mkdir logs

# Test case 1: Each SM is sampled in a round-robin way
./run.sh 1 32 > logs/case_1_1_32
grep totalSamples logs/case_1_1_32
./run.sh 1 128 > logs/case_1_1_128
grep totalSamples logs/case_1_1_128

# Test case 2: different SMs are sampled simultaneously
./run.sh 1 32 > logs/case_2_1_32
grep totalSamples logs/case_2_1_32
./run.sh 4 32 > logs/case_2_4_32
grep totalSamples logs/case_2_4_32
