cd lammps
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DPKG_KOKKOS=ON -DKOKKOS_ARCH="Power9;Volta70" -DKOKKOS_ENABLE_CUDA=yes -DKOKKOS_ENABLE_OPENMP=yes -DCMAKE_CXX_COMPILER=`pwd`/../lib/kokkos/bin/nvcc_wrapper ../cmake
make -j16
# mpirun -np 2 ./lmp -k on g 2 -sf kk -in ../src/USER-INTEL/TEST/in.intel.lj
