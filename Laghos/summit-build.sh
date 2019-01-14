export MPI_HOME=/autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/gcc-6.4.0/spectrum-mpi-10.2.0.10-20181214-o4r7cptz5i4og4c5x6ngsacosjqeo6l7/
export CPLUS_INCLUDE_PATH=`pwd`/cub-1.8.0:$CPLUS_INCLUDE_PATH

# hypre
wget https://computation.llnl.gov/projects/hypre-scalable-linear-solvers-multigrid-methods/download/hypre-2.11.2.tar.gz
tar xzvf hypre-2.11.2.tar.gz
 cd hypre-2.11.2/src
./configure --disable-fortran --with-MPI --with-MPI-include=$MPI_HOME/include --with-MPI-lib-dirs=$MPI_HOME/lib
make -j
cd ../..

# metis
wget http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-5.1.0.tar.gz
tar xzvf metis-5.1.0.tar.gz
cd metis-5.1.0
make config prefix=`pwd`
make && make install
cd ..

# mfem
git clone git@github.com:mfem/mfem.git
cd mfem
git checkout laghos-v2.0
make config MFEM_USE_MPI=YES HYPRE_DIR=`pwd`/../hypre-2.11.2/src/hypre MFEM_USE_METIS_5=YES METIS_DIR=`pwd`/../metis-5.1.0
make status
make -j
cd ..

# CUB
wget https://github.com/NVlabs/cub/archive/cub-v1.8.0.tar.gz
tar xf v1.8.0.tar.gz

# RAJA
git clone --recursive https://github.com/llnl/raja.git
cd raja
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=../../raja-install -DENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo ../
make
cd ../..

# CUDA Laghos
cd Laghos/cuda
make NV_ARCH=-arch=sm_70 CUDA_DIR=/sw/summit/cuda/9.2.148/ MPI_HOME=$MPI_HOME
cd ../..

# RAJA Laghos
cd Laghos/raja
make NV_ARCH=-arch=sm_70 CUDA_DIR=/sw/summit/cuda/9.2.148/ MPI_HOME=$MPI_HOME RAJA_DIR=`pwd`/../../raja-install
