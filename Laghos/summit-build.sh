# Tested for GCC >= 6.4.0
export MPI_HOME=`echo $MPI_HOME`
export CUDA_HOME=`echo $CUDA_HOME`
export CPLUS_INCLUDE_PATH=`pwd`/cub-1.8.0:$CPLUS_INCLUDE_PATH

# hypre
wget https://computation.llnl.gov/projects/hypre-scalable-linear-solvers-multigrid-methods/download/hypre-2.11.2.tar.gz
tar xzvf hypre-2.11.2.tar.gz
cd hypre-2.11.2/src
./configure --disable-fortran --with-MPI --with-MPI-include=$MPI_HOME/include --with-MPI-lib-dirs=$MPI_HOME/lib
make -j8
cd ../..

# metis
wget http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-5.1.0.tar.gz
tar xzvf metis-5.1.0.tar.gz
cd metis-5.1.0
make config prefix=`pwd`
make -j8 && make install
cd ..

# mfem
git clone git@github.com:mfem/mfem.git
cd mfem
git checkout laghos-v2.0
make config MFEM_DEBUG=YES MFEM_USE_MPI=YES HYPRE_DIR=`pwd`/../hypre-2.11.2/src/hypre MFEM_USE_METIS_5=YES METIS_DIR=`pwd`/../metis-5.1.0
make status
make -j8
cd ..

# CUB
wget https://github.com/NVlabs/cub/archive/cub-v1.8.0.tar.gz
tar xf v1.8.0.tar.gz

# MFEM with OCCA
git clone git@github.com:mfem/mfem.git mfem-occa
cd mfem-occa
git checkout occa-dev
make config MFEM_USE_MPI=YES HYPRE_DIR=`pwd`/../hypre-2.11.2/src/hypre MFEM_USE_METIS_5=YES METIS_DIR=`pwd`/../metis-5.1.0 MFEM_USE_OCCA=YES OCCA_DIR=`pwd`/../occa
make status
make -j8
cd ..

# CUDA Laghos
cd Laghos/cuda
make debug NV_ARCH=-arch=sm_70 CUDA_DIR=$CUDA_HOME MPI_HOME=$MPI_HOME -j8
cd ../..
