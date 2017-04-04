#include <string>
#include <vector>

#ifdef NMC_HAVE_GPU
    #include <cuda_runtime.h>
#endif

#include "memory_meter.hpp"
#include <communication/global_policy.hpp>

namespace nest {
namespace mc {
namespace util {

std::string memory_meter::name() {
    return "memory";
}

void memory_meter::take_reading() {
    readings_.push_back(allocated_memory());
}

std::vector<measurement> memory_meter::measurements() {
    using gcom = communication::global_policy;

    // Calculate the elapsed time on the local domain for each interval,
    // and store them in the times vector.
    std::vector<memory_size_type> allocated;
    allocated.push_back(0);
    for (auto i=1u; i<readings_.size(); ++i) {
        allocated.push_back(readings_[i] - readings_[i-1]);
    }

    // Assert that the same number of readings were taken on every domain.
    const auto num_readings = allocated.size();
    if (gcom::min(num_readings)!=gcom::max(num_readings)) {
        throw std::out_of_range(
            "the number of checkpoints in the \"memory\" meter do not match across domains");
    }

    // Gather the timers from across all of the domains onto the root domain.
    // Note: results are only valid on the root domain on completion.
    measurement results;
    results.name = "memory-allocated";
    results.units = "kB";
    for (auto m: allocated) {
        results.measurements.push_back(gcom::gather(std::round(m/1e3), 0));
    }

    return {results};
}

} // namespace util
} // namespace mc
} // namespace nest
