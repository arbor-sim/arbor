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

class meter_manager {
    using meter_ptr = std::unique_ptr<meter>;
    std::vector<meter_ptr> meters_;
    std::vector<std::string> checkpoint_names_;

    nlohmann::json as_json();

public:

    meter_manager();
    void checkpoint(std::string name);
    void save_to_file(const std::string& name);
};

} // namespace util
} // namespace mc
} // namespace nest
