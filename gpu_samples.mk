CXX ?= nvcc

CPU_FLAGS += -g -fopenmp

GPU_FLAGS += -O3 -lineinfo -arch $(ARCH) 

CUBIN_FLAGS += -cubin

DEVICE_FLAGS += -dc

LDFLAGS += -lcudart -lcuda -lstdc++ -lm

ifneq ($(DEVICE),)
DEVICE_OBJ := $(addsuffix .o, $(DEVICE))
endif

ifneq ($(CUBIN),)
CUBIN_CUBIN := $(addsuffix .cubin, $(CUBIN))
endif

ifeq ($(CPU_ARCH), power8)
	CPU_FLAGS += -mcpu=power8 -mtune=power8
else
ifeq ($(CPU_ARCH), power9)
	CPU_FLAGS += -mcpu=power9 -mtune=power9
endif
endif

ifdef USE_MPI
	CPU_FLAGS += -DUSE_MPI
endif

CXXFLAGS := -Xcompiler "$(CPU_FLAGS)" $(GPU_FLAGS)

all: exec cubin device
	
exec: $(EXEC)

$(EXEC): % : %.cu $(DEVICE_OBJ)
	$(CXX) -o $@ $^ $(SHOWFLAGS) $(CXXFLAGS) $(LDFLAGS)

cubin: $(CUBIN_CUBIN)

$(CUBIN_CUBIN): %.cubin : %.cu
	$(CXX) $(SHOWFLAGS) $(GPU_FLAGS) $(CUBIN_FLAGS) $<

device: $(DEVICE_OBJ)

$(DEVICE_OBJ): %.o : %.cu
	$(CXX) $(SHOWFLAGS) $(GPU_FLAGS) $(DEVICE_FLAGS) $<

clean:
	rm -rf $(EXEC) *.o *.dot *.hpcstruct *.cubin

print-% : ; $(info $* is $(flavor $*) variable set to [$($*)]) @true
