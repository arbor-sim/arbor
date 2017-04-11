#pragma once

#include <string>
#include <vector>

#include <util/power.hpp>

#include "meter.hpp"

namespace nest {
namespace mc {
namespace util {

class power_meter: public meter {
    std::vector<energy_size_type> readings_;

public:
    std::string name() override;
    std::string units() override;
    std::vector<double> measurements() override;

    void take_reading() override;
};

meter_ptr make_power_meter();

} // namespace util
} // namespace mc
} // namespace nest
