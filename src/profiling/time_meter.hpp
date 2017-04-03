#pragma once

#include <string>
#include <vector>
#include <json/json.hpp>

#include "meter.hpp"
#include "profiler.hpp"

namespace nest {
namespace mc {
namespace util {

class time_meter : public meter {
    std::vector<timer_type::time_point> readings_;

public:
    std::string name() override;
    void take_reading() override;
    virtual std::vector<measurement> measurements() override;
};

} // namespace util
} // namespace mc
} // namespace nest

