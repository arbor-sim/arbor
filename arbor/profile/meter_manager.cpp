#include <arbor/profile/timer.hpp>

#include <arbor/distributed_context.hpp>
#include <arbor/profile/meter_manager.hpp>

#include "memory_meter.hpp"
#include "power_meter.hpp"

#include "algorithms.hpp"
#include "util/hostname.hpp"
#include "util/strprintf.hpp"
#include "util/rangeutil.hpp"

namespace arb {
namespace profile {

using timer_type = timer<>;
using util::strprintf;

measurement::measurement(std::string n, std::string u,
                         const std::vector<double>& readings,
                         const distributed_context* ctx):
    name(std::move(n)), units(std::move(u))
{
    // Assert that the same number of readings were taken on every domain.
    const auto num_readings = readings.size();
    if (ctx->min(num_readings)!=ctx->max(num_readings)) {
        throw std::out_of_range(
            "the number of checkpoints in the \""+name+"\" meter do not match across domains");
    }

    // Gather across all of the domains onto the root domain.
    for (auto r: readings) {
        measurements.push_back(ctx->gather(r, 0));
    }
}

meter_manager::meter_manager(const execution_context* ctx): glob_ctx_(ctx) {
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
    arb_assert(!started_);

    started_ = true;

    // take readings for the start point
    for (auto& m: meters_) {
        m->take_reading();
    }

    // Enforce a global barrier after taking the time stamp
    glob_ctx_->distributed_context_.barrier();

    start_time_ = timer_type::tic();
};


void meter_manager::checkpoint(std::string name) {
    arb_assert(started_);

    // Record the time taken on this domain since the last checkpoint
    times_.push_back(timer<>::toc(start_time_));

    // Update meters
    checkpoint_names_.push_back(std::move(name));
    for (auto& m: meters_) {
        m->take_reading();
    }

    // Synchronize all domains before setting start time for the next interval
    glob_ctx_->distributed_context_.barrier();
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

const execution_context* meter_manager::context() const {
    return glob_ctx_;
}

// Build a report of meters, for use at the end of a simulation
// for output to file or analysis.
meter_report make_meter_report(const meter_manager& manager) {
    meter_report report;

    auto ctx = manager.context();

    // Add the times to the meter outputs
    report.meters.push_back(measurement("time", "s", manager.times(), &ctx->distributed_context_));

    // Gather the meter outputs.
    for (auto& m: manager.meters()) {
        report.meters.push_back(
            measurement(m->name(), m->units(), m->measurements(), &ctx->distributed_context_));
    }

    // Gather a vector with the names of the node that each rank is running on.
    auto host = util::hostname();
    auto hosts = ctx->distributed_context_.gather(host? *host: "unknown", 0);
    report.hosts = hosts;

    // Count the number of unique hosts.
    // This is equivalent to the number of nodes on most systems.
    util::sort(hosts);
    auto num_hosts = std::distance(hosts.begin(), std::unique(hosts.begin(), hosts.end()));

    report.checkpoints = manager.checkpoint_names();
    report.num_domains = ctx->distributed_context_.size();
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
    }
    o << "\n-------------------------------------------------------------------------------------------\n";
    int cp_index = 0;
    for (auto name: report.checkpoints) {
        name.resize(20);
        o << strprintf("%-21s", name);
        for (const auto& m: report.meters) {
            if (m.name=="time") {
                std::vector<double> times = m.measurements[cp_index];
                o << strprintf("%16.3f", algorithms::mean(times));
            }
            else if (m.name.find("memory")!=std::string::npos) {
                std::vector<double> mem = m.measurements[cp_index];
                o << strprintf("%16.3f", algorithms::mean(mem)*1e-6);
            }
            else if (m.name.find("energy")!=std::string::npos) {
                std::vector<double> e = m.measurements[cp_index];
                // TODO: this is an approximation: better reduce a subset of measurements
                auto doms_per_host = double(report.num_domains)/report.num_hosts;
                o << strprintf("%16.3f", algorithms::sum(e)/doms_per_host*1e-3);
            }
        }
        o << "\n";
        ++cp_index;
    }
    return o;
}

} // namespace util
} // namespace arb
