#include <string>
#include <vector>

#include <util/config.hpp>

#include "memory_meter.hpp"

namespace nest {
namespace mc {
namespace util {

//
//  memory_meter
//

std::string memory_meter::name() {
    return "memory-allocated";
}

std::string memory_meter::units() {
    return "B";
}

void memory_meter::take_reading() {
    readings_.push_back(allocated_memory());
}

std::vector<double> memory_meter::measurements() {
    std::vector<double> diffs;

    for (auto i=1ul; i<readings_.size(); ++i) {
        diffs.push_back(readings_[i]-readings_[i-1]);
    }

    return diffs;
}

meter_ptr make_memory_meter() {
    if (not config::has_memory_measurement) {
        return nullptr;
    }
    return meter_ptr(new memory_meter());
}

//
//  gpu_memory_meter
//

std::string gpu_memory_meter::name() {
    return "gpu-memory-allocated";
}

void gpu_memory_meter::take_reading() {
    readings_.push_back(gpu_allocated_memory());
}

meter_ptr make_gpu_memory_meter() {
    if (not config::has_cuda) {
        return nullptr;
    }
    return meter_ptr(new gpu_memory_meter());
}

} // namespace util
} // namespace mc
} // namespace nest
