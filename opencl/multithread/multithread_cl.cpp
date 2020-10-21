// g++ -g -std=c++11 multithread_cl.cpp -lOpenCL -fopenmp -o multithread_cl

#include <iostream>
#include <vector>
#include <string>
#include <time.h>
#include <omp.h>
#include <math.h>

#define CL_TARGET_OPENCL_VERSION 120
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

// Compute c = a + b.
static const char source[] =
"#if defined(cl_khr_fp64)\n"
"#  pragma OPENCL EXTENSION cl_khr_fp64: enable\n"
"#elif defined(cl_amd_fp64)\n"
"#  pragma OPENCL EXTENSION cl_amd_fp64: enable\n"
"#else\n"
"#  error double precision is not supported\n"
"#endif\n"
"kernel void add(\n"
"       ulong n,\n"
"       global const double *a,\n"
"       global const double *b,\n"
"       global double *c\n"
"       )\n"
"{\n"
"    size_t i = get_global_id(0);\n"
"    if (i < n) {\n"
"		for(int rep1=0; rep1<100;rep1++){\n"
"			for(int rep2=0; rep2<150;rep2++){\n"
"		       	c[i] = (a[i] + b[i]);\n"
"			}\n"
"		}\n"
"    }\n"
"}\n";

int main(int argc, char *argv[])
{
  int thread_count = 8;
  if (argc > 1){
	thread_count = atoi(argv[1]);
  }

  const size_t N = 1 << 20;

  try
  {
	// Get list of OpenCL platforms.
	std::vector<cl::Platform> platform;
	cl::Platform::get(&platform);

	if (platform.empty())
	{
	  std::cerr << "OpenCL platforms not found." << std::endl;
	  return 1;
	}

	// Get first available CPU device which supports double precision.
	cl::Context context;
	std::vector<cl::Device> device;
	for(auto p = platform.begin(); device.empty() && p != platform.end(); p++)
	{
	  std::vector<cl::Device> pldev;

	  try
	  {
		p->getDevices(CL_DEVICE_TYPE_GPU, &pldev);

		for(auto d = pldev.begin(); device.empty() && d != pldev.end(); d++)
		{
		  if (!d->getInfo<CL_DEVICE_AVAILABLE>()) continue;
		  std::string ext = d->getInfo<CL_DEVICE_EXTENSIONS>();

		  if (ext.find("cl_khr_fp64") == std::string::npos && ext.find("cl_amd_fp64") == std::string::npos) continue;

		  device.push_back(*d);
		  context = cl::Context(device);
		}
	  }
	  catch(...)
	  {
		device.clear();
	  }
	}

	if (device.empty())
	{
	  std::cerr << "CPUs with double precision not found." << std::endl;
	  return 1;
	}

	std::cout << device[0].getInfo<CL_DEVICE_NAME>() << std::endl;

	// Create command queue.
	cl::CommandQueue queue(context, device[0]);

	// Compile OpenCL program for found device.
	cl::Program program(context, cl::Program::Sources(1, std::make_pair(source, strlen(source))));

	try
	{
	  program.build(device);
	}
	catch (const cl::Error&)
	{
	  std::cerr
		<< "OpenCL compilation error" << std::endl
		<< program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device[0])
		<< std::endl;
	  return 1;
	}

	cl::Kernel add(program, "add");

	omp_set_num_threads(thread_count);
	#pragma omp parallel
	{
	  int thread_id = omp_get_thread_num();
	  printf("thread %d calling kernel\n", thread_id);

	  // Prepare input data.
	  std::vector<double> a(N, thread_id);
	  std::vector<double> b(N, sqrt(thread_id));
	  std::vector<double> c(N);

	  // Allocate device buffers and transfer input data to device.
	  cl::Buffer A(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, a.size() * sizeof(double), a.data());
	  cl::Buffer B(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, b.size() * sizeof(double), b.data());
	  cl::Buffer C(context, CL_MEM_READ_WRITE, c.size() * sizeof(double));

	  // Set kernel parameters.
	  add.setArg(0, static_cast<cl_ulong>(N));
	  queue.enqueueWriteBuffer(A, CL_TRUE, 0, a.size() * sizeof(double), a.data());
	  queue.enqueueWriteBuffer(B, CL_TRUE, 0, b.size() * sizeof(double), b.data());
	  queue.enqueueWriteBuffer(C, CL_TRUE, 0, c.size() * sizeof(double), c.data());
	  add.setArg(1, A);
	  add.setArg(2, B);
	  add.setArg(3, C);

	  // Launch kernel on the compute device.
	  queue.enqueueNDRangeKernel(add, cl::NullRange, N, cl::NullRange);

	  // Get result back to host.
	  queue.enqueueReadBuffer(C, CL_TRUE, 0, c.size() * sizeof(double), c.data());
		
	  // Should get '3' here.
	  std::cout << c[42] << std::endl;
	}
  }
  catch (const cl::Error &err)
  {
	std::cerr
	  << "OpenCL error: "
	  << err.what() << "(" << err.err() << ")"
	  << std::endl;
	return 1;
  }
}
