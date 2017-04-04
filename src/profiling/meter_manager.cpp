#include <communication/global_policy.hpp>
#include <util/hostname.hpp>

#include "meter_manager.hpp"

#include <json/json.hpp>

namespace nest {
namespace mc {
namespace util {

meter_manager::meter_manager() {
    // add time-measurement meter
    meters.emplace_back(new time_meter());

    // add memory consumption meter
    if (has_memory_metering) {
        meters.emplace_back(new memory_meter());
    }

    // add energy consumption meter TODO
};

void meter_manager::checkpoint(std::string name) {
    // Enforce a global synchronization point the first time that the meters
    // are used, to ensure that times measured across all domains are
    // synchronised.
    if (checkpoint_names.size()==0) {
        communication::global_policy::barrier();
    }

    checkpoint_names.push_back(std::move(name));
    for (auto& m: meters) {
        m->take_reading();
    }
}

nlohmann::json to_json(const meter_manager& manager) {
    using gcom = communication::global_policy;

    // Gather the meter outputs into a json Array
    nlohmann::json meter_out;
    for (const auto& m: manager.meters) {
        for (const auto& measure: m->measurements()) {
            meter_out.push_back(to_json(measure));
        }
    }

    // Build the hostname to rank map.
    // This is a little messy, because the names must be serialized
    // so that they might be gathered on the root rank.
    // The method I use is to copy them into a fixed size char array,
    // then unpack the result.

    auto name = hostname();

    // copy the std::string into a char array for communication
    char name_array[128];
    std::copy(name.begin(), name.end(), name_array);
    name_array[name.size()] = '\0';

    // perform global gather of host names
    auto gathered = gcom::gather(name_array, 0);

    // Only the "root" process returns meter information
    if (gcom::id()==0) {
        // push the host names into a vector of strings
        std::vector<std::string> hosts;
        for (auto s: gathered) {
            hosts.push_back(std::string(s));
        }
        return {
            {"checkpoints", manager.checkpoint_names},
            {"num_domains", gcom::size()},
            {"global_model", std::to_string(gcom::kind())},
            {"meters", meter_out},
            {"hosts", hosts},
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
        fid /*<< std::setw(1)*/ << measurements << "\n";
    }
}

} // namespace util
} // namespace mc
} // namespace nest
