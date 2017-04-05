#include <string>
#include <vector>

#include "power_meter.hpp"

namespace nest {
namespace mc {
namespace util {

std::string power_meter::name() {
    return "power";
}

void power_meter::take_reading() {
    readings_.push_back(energy());
}

std::vector<measurement> power_meter::measurements() {
    auto diff = [](energy_size_type f, energy_size_type s) {return s-f;};
    return { impl::collate(readings_, "energy", "J", diff) };
}

} // namespace util
} // namespace mc
} // namespace nest

