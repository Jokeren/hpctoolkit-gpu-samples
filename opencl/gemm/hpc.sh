# g++ -g -lOpenCL -std=c++11 gemm.cpp -o gemm
rm -r m cs gs d GTPIN_PROFILE_* 1* 
hpcrun -dd OPENCL -e gpu=opencl -t -o m ./gemm gemm.cl
# objcopy -F elf64-little --add-section .SHT_OPENCL_DEV_DEBUG=opencl_main.debuginfo opencl_main.gpubin
hpcstruct --gpucfg yes m/intel/*.gpubin -o gs
hpcstruct gemm -o cs
hpcprof -o d -S gs -S cs m/
