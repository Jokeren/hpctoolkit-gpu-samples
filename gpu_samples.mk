CXX ?= nvcc

CPU_FLAGS := -g -fopenmp

GPU_FLAGS := -lineinfo -arch $(ARCH)

ifeq ($(CPU_ARCH), power8)
	CPU_FLAGS += -mcpu=pwr8 -mtune=pwr8
else
ifeq ($(CPU_ARCH), power9)
	CPU_FLAGS += -mcpu=pwr9 -mtune=pwr9
endif
endif

ifdef USE_MPI
	CPU_FLAGS += -DUSE_MPI
endif

CXXFLAGS := -Xcompiler "$(CPU_FLAGS)" $(GPU_FLAGS)

all: exec cubin
	
exec: $(EXEC)

$(EXEC): % : %.cu
	$(CXX) $(LDFLAGS) -o $@ $< $(SHOW_FLAG) $(CXXFLAGS) -lcudart -lcuda -lstdc++ -lm

cubin: $(CUBIN)

$(CUBIN): % : %.cu
	$(CXX) $(GPU_FLAGS) -cubin $<

clean:
	rm -rf $(EXEC) *.o *.dot *.hpcstruct *.cubin
