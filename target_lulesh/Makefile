# Default build suggestion of MPI + OPENMP with Clang on IBM (Power 8) + NVIDIA GPU machines.
# You might have to change the compiler name and flags.

SHELL = /bin/sh
.SUFFIXES: .cc .o

LULESH_EXEC = lulesh2.0

MPI_INC = /opt/local/include/openmpi
MPI_LIB = /opt/local/lib
CUDA ?= /sw/summitdev/cuda/9.0.69

# Point your mpicc to Clang
CXX = clang

SOURCES2.0 = \
	lulesh.cc \
	lulesh-comm.cc \
	lulesh-viz.cc \
	lulesh-util.cc \
	lulesh-init.cc
OBJECTS2.0 = $(SOURCES2.0:.cc=.o)

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

USE_MPI = 0
mpi = 
ifdef USE_MPI
mpi = -DUSE_MPI=$(USE_MPI)
endif

showflag =
oflag = -g
archflag =
PTX = -Xcuda-ptxas "-maxrregcount=64"

# Tuning flags for Power 8
CXXFLAGS = $(archflag) -fopenmp=libomp $(oflag) -fopenmp-targets=nvptx64-nvidia-cuda $(shared) $(mpi) $(teams) $(threads) $(gpu) $(showflag) $(PTX)

LDFLAGS = -L $(CUDA)/nvvm/libdevice

.cc.o: lulesh.h
	@echo "Building $<"
	$(CXX) -c $(CXXFLAGS) -o $@  $<

all: $(LULESH_EXEC)

lulesh2.0: $(OBJECTS2.0)
	@echo "Linking"
	$(CXX) -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda $(OBJECTS2.0) $(LDFLAGS) -lomp -lomptarget -lstdc++ -lm -o $@ $(showflag)

clean:
	/bin/rm -f *.o *~ *.tgt* $(OBJECTS) $(LULESH_EXEC)
	/bin/rm -rf *.dSYM

tar: clean
	cd .. ; tar cvf lulesh-2.0.tar LULESH-2.0 ; mv lulesh-2.0.tar LULESH-2.0

