rm -r m cs gs d GTPIN_PROFILE_* 1* 
hpcrun -dd OPENCL -e gpu=opencl -t -o m ./BitonicSort
# hpcstruct --gpucfg yes m/intel/*.gpubin -o gs
hpcstruct BitonicSort -o cs
hpcprof -o d -S gs -S cs m/
