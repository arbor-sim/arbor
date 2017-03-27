#include "meter_manager.hpp"

namespace nest {
namespace mc {
namespace util {

using nlohmann::json;

meter_manager::meter_manager() {
    // add time-measurement meter
    //meters_.push_back(make_unique<meter>());
    meters_.emplace_back(new time_meter());

    // add memory consumption meter
    // TODO

    // add energy consumption meter
    // TODO
};

void meter_manager::checkpoint(std::string name) {
    checkpoint_names_.push_back(std::move(name));

    for (auto& m: meters_) {
        m->take_reading();
    }
}

json meter_manager::as_json() {
    using gcom = communication::global_policy;

    auto meter_out = json{};
    for (const auto& m: meters_) {
        meter_out.push_back(m->as_json());
    }

    // only the "root" process returns meter information
    if (!gcom::id()) {
        return {
            {"checkpoints", checkpoint_names_},
            {"num_domains", gcom::size()},
            {"meters", meter_out},
            {"global_model", std::to_string(gcom::kind())},
            // TODO mapping of domains to nodes
        };
    }

    return {};
}

void meter_manager::save_to_file(const std::string& name) {
    if (!communication::global_policy::id()) {
        std::ofstream fid;
        fid.exceptions(std::ios_base::badbit | std::ios_base::failbit);
        fid.open(name);
        fid << std::setw(2) << as_json() << "\n";
    }
}

} // namespace util
} // namespace mc
} // namespace nest
