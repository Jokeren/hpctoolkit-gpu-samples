CONFIGS := Makefile.config

include $(CONFIGS)

EXEC := main

DEVICE := vecAdd

DEVICE_OBJ := $(addprefix $(DEVICE), .o)

CXX ?= nvcc

CPU_FLAGS := -g -fopenmp

GPU_FLAGS := -lineinfo -arch $(ARCH)

ifeq ($(CPU_ARCH), power8)
	CPU_FLAGS += -mcpu=pwr8 -mtune=pwr8
endif

ifdef USE_MPI
	CPU_FLAGS += -DUSE_MPI
endif

CXXFLAGS := -Xcompiler "$(CPU_FLAGS)" $(GPU_FLAGS)

LDFLAGS := -lcudart -lcuda -lstdc++ -lm -lcudadevrt

all: exec device
	
exec: $(EXEC)

$(EXEC): % : %.cu $(DEVICE_OBJ)
	$(CXX) $(LDFLAGS) -o $@ $^ $(SHOW_FLAG) $(CXXFLAGS)

device: $(DEVICE_OBJ)

$(DEVICE_OBJ): %.o : %.cu
	$(CXX) $(GPU_FLAGS) -dc $<

clean:
	rm -rf tmp $(EXEC) *.o *.dot *.hpcstruct *.cubin

#utils
print-% : ; $(info $* is $(flavor $*) variable set to [$($*)]) @true
