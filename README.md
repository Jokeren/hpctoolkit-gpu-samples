# hpctoolkit-gpu-samples

Test cases to validate the correctness of hpctoolkit for GPU-accelerated applications.

## Usage

**Clone**

    git clone --recursive https://github.com/Jokeren/hpctoolkit-gpu-samples

**Setup**

    export OMP_NUM_THREADS=<#threads>
    export HPCTOOLKIT_GPU_TEST_REP=<#repeat times>

**Run**

    cd <sample path>
    make ARCH=<GPU arch>
    ./<application name> <device id (default 0)>

## Launch Patterns

|**Case**                         |**Purpose**                      |
|-----------------------------|-----------------------------|
|*cuda_vec_add* | *cudaLaunchKernel*|
|*cuda_cooperative_group*| *cudaLaunchCooperativeKernel* |
|*cu_vec_add* | *cuLaunchKernel*|
|*cu_multi_entries* | *cuLaunchKernel* for difference kernels with the same calling context |
|*cu_cooperative_group*| *cuLaunchCooperativeKernel* (ERROR) |
|*target_vec_add* | *omp target* |

## Call Trees

|**Case**                         |**Purpose**                      |
|-----------------------------|-----------------------------|
|*cu_call_path* | acyclic call graph |
|*cu_call_path_recursive* | recursive device function calls |
|*cu_call_path_recursive_mutual* | mutual recursive device function calls |
|*cuda_call_path_dynamic* | dynamic parallelism |
|*cuda_call_path_dynamic_recursive* | recursive call with dynamic parallelism |

## Bug Reports

|**Case**                         |**Purpose**                      |
|-----------------------------|-----------------------------|
|*nvdisasm* | nvdisasm correctness check samples |
|*cuobjdump* | cuobjdump correctness check samples |
|*cupti_test* | cupti\_test correctness check samples |

## Verification

|**Case**                         |**Purpose**                      |
|-----------------------------|-----------------------------|
|*cuda_pc_sampling_tuning* | pc sampling is performed on all SMs independently |
|*cuda_shared_memory_stall* | no stall reason indicates shared memory latency |

## Applications

|**Case**                         |**Purpose**                      | **URL** |
|-----------------------------|-----------------------------|----|
|*Laghos*| large-scale application; *RAJA* and *CUDA* programming model comparison |https://github.com/CEED/Laghos|
|*target_lulesh* | *OMP Target* performance |https://computation.llnl.gov/projects/co-design/lulesh|
|*RAJAPerf* | *CUDA* and *RAJA* performance test suite |https://github.com/LLNL/RAJAPerf|
|*sw4* | 3-D seismic modeling |https://github.com/geodynamics/sw4|
|*cuda_tensor_contraction*| nekbone | https://nek5000.mcs.anl.gov/|
|*cuda_tensor_transpose*| ExaTENSOR | https://iadac.github.io/projects/|
|*target_tensor_transpose*| *OMP Target* version of ExaTENSOR | https://iadac.github.io/projects/|
