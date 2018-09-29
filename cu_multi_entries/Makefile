SHELL = /bin/sh

EXEC = main

CUBIN = vecAdd vecSub

CXX = nvcc

USE_MPI =
mpi =
# Tuning flags for MPI
ifdef USE_MPI
mpi = -DUSE_MPI=$(USE_MPI)
endif

USE_POWER =
arch = 
# Tuning flags for Power 8
ifdef USE_POWER
archflag ?= -mcpu=pwr8 -mtune=pwr8 
endif

showflag =
oflag = -g
cuda_arch = sm_70

CXXFLAGS = -Xcompiler "$(oflag) $(arch) $(mpi) $(showflag) -fopenmp" -lcudart -lcuda $(oflag)

all: exec cubin
	
exec: $(EXEC)

$(EXEC): % : %.cu
	$(CXX) $(LDFLAGS) -lstdc++ -lm -o $@ $< $(showflag) $(CXXFLAGS)

cubin: $(CUBIN)

$(CUBIN): % : %.cu
	$(CXX) -arch $(cuda_arch) -cubin $< -lineinfo 

clean:
	/bin/rm -f *.o *~ *.tgt* $(EXEC)
	/bin/rm -rf *.dSYM
	/bin/rm -rf tmp *.dot *.hpcstruct *.cubin
	/bin/rm -rf workspace
