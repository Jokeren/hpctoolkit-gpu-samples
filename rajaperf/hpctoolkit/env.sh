#cuda
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LIBRARY_PATH
export C_INCLUDE_PATH=/usr/local/cuda-9.0/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=/usr/local/cuda-9.0/include:$CPLUS_INCLUDE_PATH

#cupti
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/extras/CUPTI/lib64:$LD_LIBRARY_PATH 
export LIBRARY_PATH=/usr/local/cuda-9.0/extras/CUPTI/lib64:$LIBRARY_PATH
export C_INCLUDE_PATH=/usr/local/cuda-9.0/extras/CUPTI/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=/usr/local/cuda-9.0/extras/CUPTI/include:$CPLUS_INCLUDE_PATH

