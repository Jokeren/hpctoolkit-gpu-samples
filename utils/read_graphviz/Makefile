
SHELL = /bin/sh
.SUFFIXES: .cc .o

EXEC = main

CXX = g++

SOURCES = \
	main.cc

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
CXXFLAGS = -lboost_graph $(oflag)

all: $(EXEC)

main: $(OBJECTS)
	@echo "Linking"
	$(CXX) $(OBJECTS) $(LDFLAGS) -lstdc++ -lm -o $@ $(showflag) $(CXXFLAGS)

clean:
	/bin/rm -f *.o *~ *.tgt* $(EXEC)
	/bin/rm -rf *.dSYM
