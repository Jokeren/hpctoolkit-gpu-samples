rm -r m cs d 
hpcrun -dd OPENCL -e gpu=opencl -o m ./multithread_cl 8
hpcstruct -o cs multithread_cl
hpcprof -o d -S cs m/
# mpirun -np 1 hpcprof-mpi -S multithread_cl.hpcstruct --metric-db yes hpctoolkit-multithread_cl-measurements/
# mpirun -np 1 hpcprof-mpi -o d -S cs m
