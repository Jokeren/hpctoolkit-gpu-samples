# cmake > 3.6
# blas
# clang
# gcc/6.4.0
# small ./bin/miniqmc_sync_move
# medium ./bin/miniqmc_sync_move -g "2 2 1"

cd miniqmc/build
GCC_HOME=$GCC_HOME
CUDA_HOME=$CUDA_HOME
BLAS_HOME=$BLAS_HOME

cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_FLAGS="--gcc-toolchain=$GCC_HOME --cuda-path=$CUDA_HOME" -DCMAKE_FIND_ROOT_PATH=$BLAS_HOME -DENABLE_OFFLOAD=0 ..
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_FLAGS="--gcc-toolchain=$GCC_HOME --cuda-path=$CUDA_HOME" -DCMAKE_FIND_ROOT_PATH=$BLAS_HOME -DENABLE_OFFLOAD=1 ..
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_FLAGS="--gcc-toolchain=$GCC_HOME --cuda-path=$CUDA_HOME"  -DENABLE_OFFLOAD=1 -DLAPACK_openblas_LIBRARY:FILEPATH=$BLAS_HOME/lib/libopenblas.a -DBLAS_openblas_LIBRARY:FILEPATH=$BLAS_HOME/lib/libopenblas.a ..
make -j16
