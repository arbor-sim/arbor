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
    time_meter() = default;

    // Every meter type should provide a human readable name
    std::string name() override;

    // records the time
    void take_reading() override;

    // This call may perform expensive operations to process and analyse the readings
    virtual nlohmann::json as_json() override;

    ~time_meter() = default;
};

} // namespace util
} // namespace mc
} // namespace nest

