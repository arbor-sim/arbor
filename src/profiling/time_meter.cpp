#include <string>
#include <vector>

#ifdef NMC_HAVE_GPU
    #include <cuda_runtime.h>
#endif

#include "time_meter.hpp"
#include <communication/global_policy.hpp>

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
    using gcom = communication::global_policy;

    // Calculate the elapsed time on the local domain for each interval,
    // and store them in the times vector.
    std::vector<double> times;
    times.push_back(0);
    for (auto i=1u; i<readings_.size(); ++i) {
        double t = timer_type::difference(readings_[i-1], readings_[i]);
        times.push_back(t);
    }

    // Assert that the same number of readings were taken on every domain.
    const auto num_readings = times.size();
    if (gcom::min(num_readings)!=gcom::max(num_readings)) {
        throw std::out_of_range(
            "the number of checkpoints in the \"time\" meter do not match across domains");
    }

    // Gather the timers from accross all of the domains onto the root domain.
    // Note: results are only valid on the root domain on completion.
    measurement results;
    results.name = "walltime";
    results.units = "s";
    for (auto t: times) {
        results.measurements.push_back(gcom::gather(t, 0));
    }

    return {results};
}

} // namespace util
} // namespace mc
} // namespace nest
