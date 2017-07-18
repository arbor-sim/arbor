#ifdef NMC_HAVE_GPU
    #include <cuda_runtime.h>
#endif

namespace nest {
namespace mc {
namespace hw {

#ifdef NMC_HAVE_GPU
unsigned num_available_gpus() {
    int n;
    cudaGetDeviceCount(&n);
    return n;
}
#else
unsigned num_available_gpus() {
    return 0u;
}
#endif

} // namespace hw
} // namespace mc
} // namespace nest
