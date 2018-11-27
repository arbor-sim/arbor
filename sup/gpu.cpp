#include <vector>

#include <arbor/version.hpp>

#ifdef ARB_GPU_ENABLED
#include <cuda_runtime.h>
#endif

#ifdef ARB_MPI_ENABLED
#include <mpi.h>
#endif

#include <sup/gpu.hpp>

namespace sup {

#ifndef ARB_GPU_ENABLED
// When arbor does not have CUDA support, return -1, which always
// indicates that no GPU is available.
int find_gpu() {
    return -1;
}

#ifdef ARB_MPI_ENABLED
int find_gpu(MPI_Comm comm) {
    return -1;
}
#endif

#else // GPU support is enabled in Arbor

int find_gpu() {
    int n;
    if (cudaGetDeviceCount(&n)==cudaSuccess) {
        // if 1 or more GPUs, take the first one.
        // else return -1 -> no gpu.
        return n? 0: -1;
    }
    return -1;
}

#ifdef ARB_MPI_ENABLED
// just a placeholder for now.
// greedy search for first available GPU.
int find_gpu(MPI_Comm comm) {
    return find_gpu();
}
#endif

#endif // ARB_GPU_ENABLED

} // namespace sup


