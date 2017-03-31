#include "meter_manager.hpp"

namespace nest {
namespace mc {
namespace util {

using nlohmann::json;

meter_manager::meter_manager() {
    // add time-measurement meter
    meters_.emplace_back(new time_meter());

    // add memory consumption meter
    // TODO

    // add energy consumption meter
    // TODO
};

void meter_manager::checkpoint(std::string name) {
    checkpoint_names_.push_back(std::move(name));

    // Enforce a global synchronization point the first time that the meters
    // are used, to ensure that times measured across all domains are
    // synchronised.
    if (meters_.size()==0) {
        communication::global_policy::barrier();
    }

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

    // Only the "root" process returns meter information
    if (gcom::id()==0) {
        return {
            {"checkpoints", checkpoint_names_},
            {"num_domains", gcom::size()},
            {"global_model", std::to_string(gcom::kind())},
            {"meters", meter_out},
            // TODO mapping of domains to nodes, which will be required to
            // calculate the total memory and energy consumption of a
            // distributed simulation.
        };
    }

    return {};
}

void meter_manager::save_to_file(const std::string& name) {
    auto measurements = as_json();
    if (!communication::global_policy::id()) {
        std::ofstream fid;
        fid.exceptions(std::ios_base::badbit | std::ios_base::failbit);
        fid.open(name);
        fid << std::setw(2) << measurements << "\n";
    }
}

} // namespace util
} // namespace mc
} // namespace nest
