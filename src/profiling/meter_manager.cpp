#include <communication/global_policy.hpp>
#include <util/hostname.hpp>
#include <util/strprintf.hpp>
#include <util/rangeutil.hpp>
#include <json/json.hpp>

#include "meter_manager.hpp"
#include "memory_meter.hpp"
#include "power_meter.hpp"

namespace arb {
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

// Build a report of meters, for use at the end of a simulation
// for output to file or analysis.
meter_report make_meter_report(const meter_manager& manager) {
    meter_report report;

    using gcom = communication::global_policy;

    // Add the times to the meter outputs
    report.meters.push_back(measurement("time", "s", manager.times()));

    // Gather the meter outputs into a json Array
    for (auto& m: manager.meters()) {
        report.meters.push_back(
            measurement(m->name(), m->units(), m->measurements()));
    }

    // Gather a vector with the names of the node that each rank is running on.
    auto host = hostname();
    report.hosts = gcom::gather(host? *host: "unknown", 0);

    report.checkpoints = manager.checkpoint_names();
    report.num_domains = gcom::size();
    report.communication_policy = gcom::kind();

    return report;
}

nlohmann::json to_json(const meter_report& report) {
    return {
        {"checkpoints", report.checkpoints},
        {"num_domains", report.num_domains},
        {"global_model", std::to_string(report.communication_policy)},
        {"meters", util::transform_view(report.meters, [](measurement const& m){return to_json(m);})},
        {"hosts", report.hosts},
    };
}

// Print easy to read report of meters to a stream.
std::ostream& operator<<(std::ostream& o, const meter_report& report) {
    o << "\n---- meters ------------------------------------------------------------\n";
    o << strprintf("%21s", "");
    for (auto const& m: report.meters) {
        if (m.name=="time") {
            o << strprintf("%16s", "time (s)");
        }
        else if (m.name.find("memory")!=std::string::npos) {
            o << strprintf("%16s", m.name+" (MB)");
        }
    }
    o << "\n------------------------------------------------------------------------\n";
    int cp_index = 0;
    for (auto name: report.checkpoints) {
        name.resize(20);
        o << strprintf("%-21s", name);
        for (const auto& m: report.meters) {
            if (m.name=="time") {
                std::vector<double> times = m.measurements[cp_index];
                o << strprintf("%16.4f", algorithms::mean(times));
            }
            else if (m.name.find("memory")!=std::string::npos) {
                std::vector<double> mem = m.measurements[cp_index];
                o << strprintf("%16.4f", algorithms::mean(mem)*1e-6);
            }
        }
        o << "\n";
        ++cp_index;
    }
    return o;
}

} // namespace util
} // namespace arb
