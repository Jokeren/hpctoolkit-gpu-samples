# hpctoolkit-gpu-samples

Test cases to validate the correctness of hpctoolkit for GPU-accelerated applications.

## Usage

**Setup**

    export OMP_NUM_THREADS = <#threads>
    export HPCTOOLKIT_GPU_TEST_REP = <#repeat times>

**Run** (except applications)

    cd <sample path>
    make ARCH=<GPU arch>
    ./<application name> <device id (default 0)>

## Launch Patterns

|**Case**                         |**Purpose**                      |
|-----------------------------|-----------------------------|
|*cuda_vec_add* | *cudaLaunchKernel*|
|*cuda_dynamic* | *cudaLaunchCooperativeKernel* |
|*cu_vec_add* | *cuLaunchKernel*|
|*cu_dynamic* | *cuLaunchCooperativeKernel*|
|*cu_multi_entries* | *cuLaunchKernel* for difference kernels with the same calling context |
|*target_vec_add* | *omp target* |

## Call Trees

|**Case**                         |**Purpose**                      |
|-----------------------------|-----------------------------|
|*cu_call_path* | acyclic call graph |
|*cu_call_path_recursive* | recursive device function calls |
|*cu_dynamic_recursive* | recursive call with dynamic parallelism |

## Bug Reports

|**Case**                         |**Purpose**                      |
|-----------------------------|-----------------------------|
|*nvdisasm* | unable to parse warp sychronization instructions |
|*cuobjdump* | incorrect ordering of dual-issued instructions |

## Applications

|**Case**                         |**Purpose**                      | **URL** |
|-----------------------------|-----------------------------|----|
|*target_lulesh* | *omp target* performance |https://computation.llnl.gov/projects/co-design/lulesh|
|*rajaperf* | *cuda* and *raja* perforance |https://github.com/LLNL/RAJAPerf|
|*sw4* | realworld application with complex call trees |https://github.com/geodynamics/sw4|

