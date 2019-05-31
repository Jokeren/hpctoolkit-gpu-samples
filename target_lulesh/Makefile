TEMPLATE := ../gpu_samples_target.mk

EXEC := lulesh

SOURCES := \
	lulesh.cc \
	lulesh-comm.cc \
	lulesh-viz.cc \
	lulesh-util.cc \
	lulesh-init.cc

USE_GPU := 1

USE_MPI := 0

ARCHFLAGS := -mcpu=pwr9 -mtune=pwr9

OMPFLAGS := -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda

CXX := clang++

include $(TEMPLATE)
