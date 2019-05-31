TEMPLATE := ../gpu_samples_target.mk

EXEC := main

SOURCES := main.cc

USE_GPU := 1

ARCHFLAGS := -mcpu=pwr9 -mtune=pwr9

OMPFLAGS := -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda

CXX := clang++

SHOWFLAGS :=

include $(TEMPLATE)
