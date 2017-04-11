#include <string>
#include <vector>

#include <util/config.hpp>

#include "power_meter.hpp"

namespace nest {
namespace mc {
namespace util {

std::string power_meter::name() {
    return "energy";
}

std::string power_meter::units() {
    return "J";
}

void power_meter::take_reading() {
    readings_.push_back(energy());
}

std::vector<double> power_meter::measurements() {
    std::vector<double> diffs;

    for (auto i=1ul; i<readings_.size(); ++i) {
        diffs.push_back(readings_[i]-readings_[i-1]);
    }

    return diffs;
}

meter_ptr make_power_meter() {
    if (not config::has_power_measurement) {
        return nullptr;
    }
    return meter_ptr(new power_meter());
}

} // namespace util
} // namespace mc
} // namespace nest
