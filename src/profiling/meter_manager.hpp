#pragma once

#include <memory>
#include <vector>

#include <util/make_unique.hpp>
#include <communication/global_policy.hpp>
#include <json/json.hpp>

#include "meter.hpp"
#include "time_meter.hpp"

namespace nest {
namespace mc {
namespace util {

struct meter_manager {
    std::vector<std::unique_ptr<meter>> meters;
    std::vector<std::string> checkpoint_names;

    meter_manager();
    void checkpoint(std::string name);
};

nlohmann::json to_json(const meter_manager&);
void save_to_file(const meter_manager& manager, const std::string& name);

} // namespace util
} // namespace mc
} // namespace nest
