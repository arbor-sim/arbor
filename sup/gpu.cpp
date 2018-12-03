#if defined(ARB_WITH_GPU)

#include <cuda_runtime.h>

namespace sup {

// When arbor does not have CUDA support, return -1, which always
// indicates that no GPU is available.
int default_gpu() {
    int n;
    if (cudaGetDeviceCount(&n)==cudaSuccess) {
        // if 1 or more GPUs, take the first one.
        // else return -1 -> no gpu.
        return n? 0: -1;
    }
    return -1;
}

} // namespace sup

#else // defined(ARB_WITH_GPU)

namespace sup {

int default_gpu() {
    return -1;
}

} // namespace sup

#endif // ARB_WITH_GPU

