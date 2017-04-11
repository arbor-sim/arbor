#include <communication/global_policy.hpp>
#include <util/hostname.hpp>
#include <json/json.hpp>

#include "meter_manager.hpp"
#include "memory_meter.hpp"
#include "power_meter.hpp"

namespace nest {
namespace mc {
namespace util {

measurement::measurement(
        std::string n, std::string u, const std::vector<double>& readings):
    name(std::move(n)), units(std::move(u))
{
    using gcom = communication::global_policy;

    // Assert that the same number of readings were taken on every domain.
    const auto num_readings = readings.size();
    if (gcom::min(num_readings)!=gcom::max(num_readings)) {
        throw std::out_of_range(
            "the number of checkpoints in the \""+name+"\" meter do not match across domains");
    }

    // Gather across all of the domains onto the root domain.
    for (auto r: readings) {
        measurements.push_back(gcom::gather(r, 0));
    }
}

meter_manager::meter_manager() {
    if (auto m = make_memory_meter()) {
        meters_.push_back(std::move(m));
    }
    if (auto m = make_gpu_memory_meter()) {
        meters_.push_back(std::move(m));
    }
    if (auto m = make_power_meter()) {
        meters_.push_back(std::move(m));
    }
};

void meter_manager::start() {
    EXPECTS(!started_);

    started_ = true;

    // take readings for the start point
    for (auto& m: meters_) {
        m->take_reading();
    }

    // Enforce a global barrier after taking the time stamp
    communication::global_policy::barrier();

    start_time_ = timer_type::tic();
};


void meter_manager::checkpoint(std::string name) {
    EXPECTS(started_);

    // Record the time taken on this domain since the last checkpoint
    auto end_time = timer_type::tic();
    times_.push_back(timer_type::difference(start_time_, end_time));

    // Update meters
    checkpoint_names_.push_back(std::move(name));
    for (auto& m: meters_) {
        m->take_reading();
    }

    // Synchronize all domains before setting start time for the next interval
    communication::global_policy::barrier();
    start_time_ = timer_type::tic();
}

const std::vector<std::unique_ptr<meter>>& meter_manager::meters() const {
    return meters_;
}

const std::vector<std::string>& meter_manager::checkpoint_names() const {
    return checkpoint_names_;
}

const std::vector<double>& meter_manager::times() const {
    return times_;
}

nlohmann::json to_json(const measurement& mnt) {
    nlohmann::json measurements;
    for (const auto& m: mnt.measurements) {
        measurements.push_back(m);
    }

    return {
        {"name", mnt.name},
        {"units", mnt.units},
        {"measurements", measurements}
    };
}

nlohmann::json to_json(const meter_manager& manager) {
    using gcom = communication::global_policy;

    // Gather the meter outputs into a json Array
    nlohmann::json meter_out;
    for (auto& m: manager.meters()) {
        meter_out.push_back(
            to_json(measurement(m->name(), m->units(), m->measurements()))
        );
    }
    // Add the times to the meter outputs
    meter_out.push_back(to_json(measurement("time", "s", manager.times())));

    // Gather a vector with the names of the node that each rank is running on.
    auto hosts = gcom::gather(hostname(), 0);

    // Only the "root" process returns meter information
    if (gcom::id()==0) {
        return {
            {"checkpoints", manager.checkpoint_names()},
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
        fid << std::setw(1) << measurements << "\n";
    }
}

} // namespace util
} // namespace mc
} // namespace nest
