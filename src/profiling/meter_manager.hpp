#pragma once

#include <memory>
#include <vector>

#include <util/make_unique.hpp>
#include <communication/global_policy.hpp>
#include <json/json.hpp>

#include "meter.hpp"
#include "memory_meter.hpp"
#include "time_meter.hpp"

namespace nest {
namespace mc {
namespace util {

class meter_manager {
    std::vector<std::unique_ptr<meter>> meters_;
    std::vector<std::string> checkpoint_names_;

public:
    meter_manager();
    void checkpoint(std::string name);

    const std::vector<std::unique_ptr<meter>>& meters() const;
    const std::vector<std::string>& checkpoint_names() const;
};

nlohmann::json to_json(const meter_manager&);
void save_to_file(const meter_manager& manager, const std::string& name);

} // namespace util
} // namespace mc
} // namespace nest
