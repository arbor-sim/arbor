#pragma once

#include <string>
#include <vector>

#include <util/power.hpp>

#include "meter.hpp"

namespace nest {
namespace mc {
namespace util {

class power_meter : public meter {
    std::vector<energy_size_type> readings_;

public:
    std::string name() override;
    void take_reading() override;
    virtual std::vector<measurement> measurements() override;
};

} // namespace util
} // namespace mc
} // namespace nest

