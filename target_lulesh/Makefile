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

ARCHFLAG := -mcpu=pwr9 -mtune=pwr9

include $(TEMPLATE)
