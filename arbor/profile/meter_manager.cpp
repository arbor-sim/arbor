#include <arbor/profile/timer.hpp>

#include <arbor/profile/meter_manager.hpp>
#include <arbor/context.hpp>

#include "memory_meter.hpp"
#include "power_meter.hpp"

#include "algorithms.hpp"
#include "execution_context.hpp"
#include "util/hostname.hpp"
#include "util/strprintf.hpp"
#include "util/rangeutil.hpp"

namespace arb {
namespace profile {

using timer_type = timer<>;
using util::strprintf;

measurement::measurement(std::string n, std::string u,
                         const std::vector<double>& readings,
                         const context& ctx):
    name(std::move(n)), units(std::move(u))
{
    auto dist = ctx->distributed;

    // Assert that the same number of readings were taken on every domain.
    const auto num_readings = readings.size();
    if (dist->min(num_readings)!=dist->max(num_readings)) {
        throw std::out_of_range(
            "the number of checkpoints in the \""+name+"\" meter do not match across domains");
    }

    // Gather across all of the domains onto the root domain.
    for (auto r: readings) {
        measurements.push_back(dist->gather(r, 0));
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

void meter_manager::start(const context& ctx) {
    arb_assert(!started_);

    started_ = true;

    // take readings for the start point
    for (auto& m: meters_) {
        m->take_reading();
    }

    // Enforce a global barrier after taking the time stamp
    ctx->distributed->barrier();

    start_time_ = timer_type::tic();
};


void meter_manager::checkpoint(std::string name, const context& ctx) {
    arb_assert(started_);

    // Record the time taken on this domain since the last checkpoint
    times_.push_back(timer<>::toc(start_time_));

    // Update meters
    checkpoint_names_.push_back(std::move(name));
    for (auto& m: meters_) {
        m->take_reading();
    }

    // Synchronize all domains before setting start time for the next interval
    ctx->distributed->barrier();
    start_time_ = timer<>::tic();
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

// Build a report of meters, for use at the end of a simulation
// for output to file or analysis.
meter_report make_meter_report(const meter_manager& manager, const context& ctx) {
    meter_report report;

    // Add the times to the meter outputs
    report.meters.push_back(measurement("time", "s", manager.times(), ctx));

    // Gather the meter outputs.
    for (auto& m: manager.meters()) {
        report.meters.push_back(
            measurement(m->name(), m->units(), m->measurements(), ctx));
    }

    // Gather a vector with the names of the node that each rank is running on.
    auto host = util::hostname();
    auto hosts = ctx->distributed->gather(host? *host: "unknown", 0);
    report.hosts = hosts;

    // Count the number of unique hosts.
    // This is equivalent to the number of nodes on most systems.
    util::sort(hosts);
    auto num_hosts = std::distance(hosts.begin(), std::unique(hosts.begin(), hosts.end()));

    report.checkpoints = manager.checkpoint_names();
    report.num_domains = ctx->distributed->size();
    report.num_hosts = num_hosts;

    return report;
}

// Print easy to read report of meters to a stream.
std::ostream& operator<<(std::ostream& o, const meter_report& report) {
    o << "\n---- meters -------------------------------------------------------------------------------\n";
    o << strprintf("meter%16s", "");
    for (auto const& m: report.meters) {
        if (m.name=="time") {
            o << strprintf("%16s", "time(s)");
        }
        else if (m.name.find("memory")!=std::string::npos) {
            o << strprintf("%16s", m.name+"(MB)");
        }
        else if (m.name.find("energy")!=std::string::npos) {
            o << strprintf("%16s", m.name+"(kJ)");
        }
        else {
            o << strprintf("%16s(avg)", m.name);
        }
    }
    o << "\n-------------------------------------------------------------------------------------------\n";
    std::vector<double> sums(report.meters.size());
    int cp_index = 0;
    for (auto name: report.checkpoints) {
        name.resize(20);
        o << strprintf("%-21s", name);
        int m_index = 0;
        for (const auto& m: report.meters) {
            if (m.name=="time") {
                // Calculate the average time per rank in s.
                double time = algorithms::mean(m.measurements[cp_index]);
                sums[m_index] += time;
                o << strprintf("%16.3f", time);
            }
            else if (m.name.find("memory")!=std::string::npos) {
                // Calculate the average memory per rank in MB.
                double mem = algorithms::mean(m.measurements[cp_index])*1e-6;
                sums[m_index] += mem;
                o << strprintf("%16.3f", mem);
            }
            else if (m.name.find("energy")!=std::string::npos) {
                auto doms_per_host = double(report.num_domains)/report.num_hosts;
                // Calculate the total energy consumed accross all ranks in kJ 
                // Energy measurements are per "per node", so only normalise
                // by the number of ranks per node. TODO, this is an
                // approximation: better reduce a subset of measurements.
                double energy = util::sum(m.measurements[cp_index])/doms_per_host*1e-3;
                sums[m_index] += energy;
                o << strprintf("%16.3f", energy);
            }
            else {
                double value = algorithms::mean(m.measurements[cp_index]);
                sums[m_index] += value;
                o << strprintf("%16.3f", value);
            }
            ++m_index;
        }
        o << "\n";
        ++cp_index;
    }

    // Print a final line with the accumulated values of each meter.
    o << strprintf("%-21s", "meter-total");
    for (const auto& v: sums) {
        o << strprintf("%16.3f", v);
    }
    o << "\n";

    return o;
}

} // namespace util
} // namespace arb
