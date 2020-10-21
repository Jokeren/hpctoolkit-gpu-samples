# opencl runtime and c++ headers needed in environment

CXX=g++
OPT=-g
CXX_VERSION=-std=c++11
LIBS=-lOpenCL

gemm: gemm.cpp
	$(CXX) $(OPT) $(CXX_VERSION) $(LIBS) -o gemm gemm.cpp

clean:
	rm gemm
