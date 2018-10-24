CONFIGS := Makefile.config

include $(CONFIGS)

EXEC := main

CUBIN := vecAdd

CXX := $(CUDA_DIR)/bin/nvcc

CPU_FLAGS := -g -fopenmp

GPU_FLAGS := -lineinfo -arch $(ARCH)

ifeq ($(CPU_ARCH), power8)
	CPU_FLAGS += -mcpu=pwr8 -mtune=pwr8
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
	rm -rf tmp $(EXEC) *.o *.hpcstruct *.cubin
