#pragma once

#include <memory>
#include <vector>

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

public:

    void checkpoint(std::string name) {
        checkpoint_names_.push_back(std::move(name));

        for (auto& m: meters_) {
            m->take_reading();
        }
    }

    nlohmann::json as_json() {
        using gcom = communication::global_policy;

        nlohmann::json meter_out = {};
        for (const auto& m: meters_) {
            meter_out.push_back(m->as_json());
        }

        nlohmann::json result = {
            {"checkpoints", checkpoint_names_},
            {"num_domains", gcom::size()},
            {"meters", meter_out},
            // number of mpi ranks
            // list of checkpoints by name
            // global mode: serial, dryrun or mpi
            // mapping of domains to nodes
        };

        return result;
    }
};

} // namespace util
} // namespace mc
} // namespace nest
