#pragma once

#include <string>
#include <json/json.hpp>

namespace nest {
namespace mc {
namespace util {

class meter {
public:
    meter() = default;

    // Every meter type should provide a human readable name
    virtual std::string name() = 0;

    virtual void take_reading() = 0;

    // This call may perform expensive operations to process and analyse the readings
    virtual nlohmann::json as_json() = 0;

    virtual ~meter() = default;
};

} // namespace util
} // namespace mc
} // namespace nest
