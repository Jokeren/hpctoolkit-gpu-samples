# hpctoolkit-gpu-samples

Test cases to validate the correctness of hpctoolkit for GPU-accelerated applications.

## Usage

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

## Applications

|**Case**                         |**Purpose**                      | **URL** |
|-----------------------------|-----------------------------|----|
|*target_lulesh* | *omp target* performance |https://computation.llnl.gov/projects/co-design/lulesh|
|*rajaperf* | *cuda* and *raja* perforance |https://github.com/LLNL/RAJAPerf|
|*sw4* | realworld application with complex call trees |https://github.com/geodynamics/sw4|

