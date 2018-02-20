
SHELL = /bin/sh
.SUFFIXES: .cc .o

EXEC = main

CUDA ?= /sw/summitdev/cuda/9.0.69

CXX = nvcc

SOURCES = \
	main.cu

OBJECTS = $(SOURCES:.cc=.o)

USE_MPI =
mpi =
ifdef USE_MPI
mpi = -DUSE_MPI=$(USE_MPI)
endif

showflag =
oflag = -g
archflag ?= -mcpu=pwr8 -mtune=pwr8 

# Tuning flags for Power 8
CXXFLAGS = $(archflag) $(oflag) $(shared) $(mpi) $(showflag) -fopenmp

.cc.o:
	@echo "Building $<"
	$(CXX) -c $(CXXFLAGS) -o $@  $<

all: $(EXEC)

main: $(OBJECTS)
	@echo "Linking"
	$(CXX) $(OBJECTS) $(LDFLAGS) -lomp -lstdc++ -lm -o $@ $(showflag)

clean:
	/bin/rm -f *.o *~ *.tgt* $(OBJECTS) $(EXEC)
	/bin/rm -rf *.dSYM
