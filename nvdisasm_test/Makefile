CUBIN = synthesis

CXX = nvcc

OFLAG = -O3

ARCH = sm_70

CXXFLAGS = -cubin -arch $(ARCH) $(OFLAG)

all: cubin
	
cubin: $(CUBIN)

$(CUBIN): % : %.cu
	$(CXX) $(CXXFLAGS) $<

clean:
	rm $(CUBIN).cubin

