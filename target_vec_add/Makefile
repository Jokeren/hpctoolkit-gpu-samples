
SHELL = /bin/sh
.SUFFIXES: .cc .o

EXEC = main

CUDA ?= /sw/summit/cuda/9.2.148

# Point your mpicc to Clang
CXX = clang

SOURCES = \
	main.cc

OBJECTS = $(SOURCES:.cc=.o)

teams = 
ifdef TEAMS
teams = -DTEAMS=$(TEAMS)
endif

threads = 
ifdef THREADS
threads = -DTHREADS=$(THREADS)
endif

USE_GPU = 1
gpu =
ifdef USE_GPU
gpu = -DUSE_GPU=$(USE_GPU) --cuda-path=$(CUDA)
endif

USE_MPI =
mpi =
ifdef USE_MPI
mpi = -DUSE_MPI=$(USE_MPI)
endif

showflag =
oflag = -g
archflag ?= -mcpu=pwr9 -mtune=pwr9

# Tuning flags for Power 8
CXXFLAGS = $(archflag) -fopenmp=libomp $(oflag) -fopenmp-targets=nvptx64-nvidia-cuda $(shared) $(mpi) $(teams) $(threads) $(gpu) $(showflag)

LDFLAGS = -L $(CUDA)/nvvm/libdevice

.cc.o:
	@echo "Building $<"
	$(CXX) --cuda-path=/sw/summit/cuda/9.2.148 -c $(CXXFLAGS) -o $@  $<

all: $(EXEC)

main: $(OBJECTS)
	@echo "Linking"
	$(CXX) -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda $(OBJECTS) $(LDFLAGS) -lomp -lomptarget -lstdc++ -lm -o $@ $(showflag)

clean:
	/bin/rm -f *.o *~ *.tgt* $(OBJECTS) $(EXEC)
	/bin/rm -rf *.dSYM
