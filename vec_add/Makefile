
SHELL = /bin/sh
.SUFFIXES: .cc .o

EXEC = main

CUDA ?= /sw/summitdev/cuda/9.0.69

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
gpu = -DUSE_GPU=$(USE_GPU)
endif

mpi = -DUSE_MPI=0
ifdef USE_MPI
mpi = -DUSE_MPI=$(USE_MPI)
endif

showflag =
oflag = -g
archflag ?= -mcpu=pwr8 -mtune=pwr8 

# Tuning flags for Power 8
CXXFLAGS = $(archflag) -fopenmp=libomp $(oflag) -fopenmp-targets=nvptx64-nvidia-cuda $(shared) $(mpi) $(teams) $(threads) $(gpu) $(showflag)

LDFLAGS = -L $(CUDA)/nvvm/libdevice

.cc.o:
	@echo "Building $<"
	$(CXX) -c $(CXXFLAGS) -o $@  $<

all: $(EXEC)

main: $(OBJECTS)
	@echo "Linking"
	$(CXX) -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda $(OBJECTS) $(LDFLAGS) -lomp -lomptarget -lstdc++ -lm -o $@ $(showflag)

clean:
	/bin/rm -f *.o *~ *.tgt* $(OBJECTS) $(EXEC)
	/bin/rm -rf *.dSYM
