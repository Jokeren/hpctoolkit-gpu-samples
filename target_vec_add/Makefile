TEMPLATE := ../gpu_samples_target.mk

EXEC := main

SOURCES := main.cc

USE_GPU := 1

ARCHFLAG := -mcpu=pwr9 -mtune=pwr9

OMPFLAG := -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda

SHOWFLAG :=

CXX := clang++

include $(TEMPLATE)
