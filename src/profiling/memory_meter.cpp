#include <string>
#include <vector>

#include "memory_meter.hpp"

namespace nest {
namespace mc {
namespace util {

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
    auto memdiff = [](memory_size_type f, memory_size_type s) { return s-f; };

    std::vector<measurement> results;

    results.push_back(impl::collate(readings_, "memory-allocated", "B", memdiff));
    if (readings_gpu_.size()) {
        results.push_back(impl::collate(readings_gpu_, "memory-allocated-gpu", "B", memdiff));
    }
    return results;
}

} // namespace util
} // namespace mc
} // namespace nest
