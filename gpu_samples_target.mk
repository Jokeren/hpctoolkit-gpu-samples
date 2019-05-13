# Default build suggestion of MPI + OPENMP with Clang on IBM (Power 8) + NVIDIA GPU machines.
# You might have to change the compiler name and flags.
# Point your mpicc to clang or xlc
CXX ?= clang++

OBJECTS := $(SOURCES:.cc=.o)

ifneq ($(TEAMS),)
teams = -DTEAMS=$(TEAMS)
endif

ifneq ($(THREADS),)
threads = -DTHREADS=$(THREADS)
endif

ifneq ($(USE_GPU),)
gpu = -DUSE_GPU=$(USE_GPU)
endif

ifneq ($(USE_MPI),)
mpi = -DUSE_MPI=$(USE_MPI)
endif

SHOWFLAG +=
OFLAG += -O3
OMPFLAG +=
ARCHFLAG += 
PTXFLAG +=

CXXFLAGS = $(ARCHFLAG) $(OMPFLAG) $(OFLAG) $(SHOWFLAG) $(PTXFLAG) $(mpi) $(teams) $(threads) $(gpu) -std=c++11

LDFLAGS +=

all: obj exec

obj: $(OBJECTS)

$(OBJECTS): %.o : %.cc
	@echo "Building $@"
	$(CXX) $(CXXFLAGS) -o $@ -c $<

exec: $(EXEC)

$(EXEC): $(OBJECTS)
	@echo "Linking $@"
	$(CXX) $(OMPFLAG) $(LDFLAGS) -o $@ $^ $(SHOWFLAG)

clean:
	/bin/rm -f *.o *~ *.tgt* *.hpcstruct* $(OBJECTS) $(EXEC)
	/bin/rm -rf *.dSYM

print-% : ; $(info $* is $(flavor $*) variable set to [$($*)]) @true
