# opencl runtime and c++ headers needed in environment

CXX=g++
OPT=-g
CXX_VERSION=-std=c++11
LIBS=-lOpenCL -fopenmp

multithread: multithread_cl.cpp
	$(CXX) $(OPT) $(CXX_VERSION) $(LIBS) -o multithread_cl multithread_cl.cpp

clean:
	rm multithread_cl
