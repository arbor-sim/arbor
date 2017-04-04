#include "meter_manager.hpp"

namespace nest {
namespace mc {
namespace util {

meter_manager::meter_manager() {
    // add time-measurement meter
    meters.emplace_back(new time_meter());

    // add memory consumption meter
    // TODO

    // add energy consumption meter
    // TODO
};

void meter_manager::checkpoint(std::string name) {
    checkpoint_names.push_back(std::move(name));

    // Enforce a global synchronization point the first time that the meters
    // are used, to ensure that times measured across all domains are
    // synchronised.
    if (meters.size()==0) {
        communication::global_policy::barrier();
    }

    for (auto& m: meters) {
        m->take_reading();
    }
}

nlohmann::json to_json(const meter_manager& manager) {
    using gcom = communication::global_policy;

    nlohmann::json meter_out;
    for (const auto& m: manager.meters) {
        for (const auto& measure: m->measurements()) {
            meter_out.push_back(to_json(measure));
        }
    }

    // Only the "root" process returns meter information
    if (gcom::id()==0) {
        return {
            {"checkpoints", manager.checkpoint_names},
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

void save_to_file(const meter_manager& manager, const std::string& name) {
    auto measurements = to_json(manager);
    if (!communication::global_policy::id()) {
        std::ofstream fid;
        fid.exceptions(std::ios_base::badbit | std::ios_base::failbit);
        fid.open(name);
        fid << std::setw(1) << measurements << "\n";
    }
}

} // namespace util
} // namespace mc
} // namespace nest
