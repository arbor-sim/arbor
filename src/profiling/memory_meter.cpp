#include <string>
#include <vector>

#include "memory_meter.hpp"
#include <communication/global_policy.hpp>

namespace nest {
namespace mc {
namespace util {

namespace {
    measurement collate(const std::vector<memory_size_type>& readings, std::string name) {
        using gcom = communication::global_policy;

        // Calculate the local change in allocated memory for each interval.
        std::vector<memory_size_type> allocated;
        allocated.push_back(0);
        for (auto i=1u; i<readings.size(); ++i) {
            allocated.push_back(readings[i] - readings[i-1]);
        }

        // Assert that the same number of readings were taken on every domain.
        const auto num_readings = allocated.size();
        if (gcom::min(num_readings)!=gcom::max(num_readings)) {
            throw std::out_of_range(
                "the number of checkpoints in the \"memory\" meter do not match across domains");
        }

        // Gather allocations from across all of the domains onto the root domain.
        // Note: results are only valid on the root domain on completion.
        measurement results;
        results.name = std::move(name);
        results.units = "kB";
        for (auto m: allocated) {
            results.measurements.push_back(gcom::gather(std::round(m/1e3), 0));
        }

        return results;
    }
} // anonymous namespace

std::string memory_meter::name() {
    return "memory";
}

void memory_meter::take_reading() {
    readings_.push_back(allocated_memory());
    #ifdef NMC_HAVE_GPU
    readings_gpu_.push_back(gpu_allocated_memory());
    #endif
}

std::vector<measurement> memory_meter::measurements() {
    std::vector<measurement> results;
    results.push_back(collate(readings_, "memory-allocated"));
    if (readings_gpu_.size()) {
        results.push_back(collate(readings_gpu_, "memory-allocated-gpu"));
    }
    return results;
}

} // namespace util
} // namespace mc
} // namespace nest
