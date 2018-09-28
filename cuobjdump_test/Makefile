CUBIN = vecAdd

CXX = nvcc

OFLAG = -O3

ARCH = sm_60

CXXFLAGS = -cubin -arch $(ARCH) $(OFLAG)

all: cubin
	
cubin: $(CUBIN)

$(CUBIN): % : %.cu
	$(CXX) $(CXXFLAGS) $<

clean:
	rm $(CUBIN).cubin

