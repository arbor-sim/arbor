#ifdef NMC_HAVE_GPU
    #include <cuda_runtime.h>
#endif

namespace arb {
namespace hw {

#ifdef NMC_HAVE_GPU
unsigned num_gpus() {
    int n;
    cudaGetDeviceCount(&n);
    return n;
}
#else
unsigned num_gpus() {
    return 0u;
}
#endif

} // namespace hw
} // namespace arb
