#include <string>
#include <vector>

#ifdef NMC_HAVE_GPU
    #include <cuda_runtime.h>
#endif

#include "time_meter.hpp"

namespace nest {
namespace mc {
namespace util {

std::string time_meter::name() {
    return "time";
}

void time_meter::take_reading() {
    // Wait for execution on this global domain to finish before recording the
    // time stamp. For now this means waiting for all work to finish executing
    // on the GPU (if GPU support is enabled)
#ifdef NMC_HAVE_GPU
    cudaDeviceSynchronize();
#endif

    // Record the time stamp
    readings_.push_back(timer_type::tic());

    // Enforce a global barrier after taking the time stamp
    communication::global_policy::barrier();
}

std::vector<measurement> time_meter::measurements() {
    return {impl::collate(readings_, "walltime", "s", timer_type::difference)};
}

} // namespace util
} // namespace mc
} // namespace nest
