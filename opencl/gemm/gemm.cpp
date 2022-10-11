//g++ -g -lOpenCL -std=c++11 gemm.cpp -o gemm

#include <iostream>
#include <vector>
#include <string>
#include <time.h>

#define CL_TARGET_OPENCL_VERSION 120
#define __CL_ENABLE_EXCEPTIONS
#include "CL/cl.hpp"

int main(int argc, char* argv[]) {
	clock_t start_t, end_t;
	start_t = clock();
	const size_t N = 1 << 10;
	
	if (argc == 1) {
		std::cerr << "Please pass the source code file. Exiting.." << std::endl;	
		exit(1);
	}
	// read a string file
	char *file_name = argv[1];
	//char *file_name = "gemm.cl";
	FILE *f = fopen(file_name, "rb");
	fseek(f, 0, SEEK_END);
	long fsize = ftell(f);
	fseek(f, 0, SEEK_SET);
	char *source = (char *)malloc(fsize + 1);
	fread(source, fsize, 1, f);
	fclose(f);
	source[fsize] = 0;

    try {
	// Get list of OpenCL platforms.
	std::vector<cl::Platform> platform;
	cl::Platform::get(&platform);

	if (platform.empty()) {
	    std::cerr << "OpenCL platforms not found." << std::endl;
	    return 1;
	}

	// Get first available GPU device which supports double precision.
	cl::Context context;
	std::vector<cl::Device> device;
	for(auto p = platform.begin(); device.empty() && p != platform.end(); p++) {
	    std::vector<cl::Device> pldev;

	    try {
		p->getDevices(CL_DEVICE_TYPE_GPU, &pldev);

		for(auto d = pldev.begin(); device.empty() && d != pldev.end(); d++) {
		    if (!d->getInfo<CL_DEVICE_AVAILABLE>()) continue;

		    std::string ext = d->getInfo<CL_DEVICE_EXTENSIONS>();

		    if (
			    ext.find("cl_khr_fp64") == std::string::npos &&
			    ext.find("cl_amd_fp64") == std::string::npos
		       ) continue;

		    device.push_back(*d);
		    context = cl::Context(device);
		}
	    } catch(...) {
		device.clear();
	    }
	}

	if (device.empty()) {
	    std::cerr << "GPUs with double precision not found." << std::endl;
	    return 1;
	}

	std::cout << device[0].getInfo<CL_DEVICE_NAME>() << std::endl;

	// Create command queue.
	cl::CommandQueue queue(context, device[0]);

	// Compile OpenCL program for found device.
	cl::Program program(context, cl::Program::Sources(
		    1, std::make_pair(source, strlen(source))
		    ));

	try {
	    program.build(device);
	} catch (const cl::Error&) {
	    std::cerr
		<< "OpenCL compilation error" << std::endl
		<< program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device[0])
		<< std::endl;
	    return 1;
	}

	cl::Kernel gemm(program, "GEMM");

	// Prepare input data.
	std::vector<double> a(N, 1);
	std::vector<double> b(N, 2);
	std::vector<double> c(N);

	// Allocate device buffers and transfer input data to device.
	cl::Buffer A(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		a.size() * sizeof(double), a.data());

	cl::Buffer B(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		b.size() * sizeof(double), b.data());

	cl::Buffer C(context, CL_MEM_READ_WRITE,
		c.size() * sizeof(double));

	/*cl::Memory temp = cl::Buffer(context, CL_MEM_READ_WRITE, 4*sizeof(float), NULL, NULL);
 	size_t mem_obj_size;
    clGetMemObjectInfo(temp, CL_MEM_SIZE, sizeof(mem_obj_size), &mem_obj_size, NULL);
	printf(" meminfo: %lu\n", mem_obj_size);*/
	
// Set kernel parameters.

	queue.enqueueWriteBuffer(A, CL_TRUE, 0, a.size() * sizeof(double), a.data());
	queue.enqueueWriteBuffer(B, CL_TRUE, 0, b.size() * sizeof(double), b.data());
	queue.enqueueWriteBuffer(C, CL_TRUE, 0, c.size() * sizeof(double), c.data());
	gemm.setArg(0, A);
	gemm.setArg(1, B);
	gemm.setArg(2, C);
	gemm.setArg(3, static_cast<int>(N));
	
	// Launch kernel on the compute device.
	queue.enqueueNDRangeKernel(gemm, cl::NullRange, N, cl::NullRange);

	// Get result back to host.
	queue.enqueueReadBuffer(C, CL_TRUE, 0, c.size() * sizeof(double), c.data());

	// Should get '3' here.
	//std::cout << c[42] << std::endl;
    } catch (const cl::Error &err) {
	std::cerr
	    << "OpenCL error: "
	    << err.what() << "(" << err.err() << ")"
	    << std::endl;
	return 1;
    }
	end_t = clock();
	printf("Execution time: %f\n", (double)(end_t - start_t) / CLOCKS_PER_SEC);
}
