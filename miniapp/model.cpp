#include <cstdlib>
#include <vector>

#include "catypes.hpp"
#include "cell.hpp"
#include "cell_group.hpp"
#include "communication/communicator.hpp"
#include "communication/global_policy.hpp"
#include "fvm_cell.hpp"
#include "profiling/profiler.hpp"
#include "threading/threading.hpp"

namespace nest {
namespace mc {

struct model {
    using cell_group = cell_group<fvm::fvm_cell<double, cell_local_size_type>>;

    void reset() {
        t_ = 0.;
        // otherwise unimplemented
        std::abort();
    }

    double run(double tuntil, double dt) {
        while (t_<tuntil) {
            auto tstep = std::min(t_+dt, tunitl);
            threading::parallel_for::apply(
                0u, cell_groups.size(),
                [&](unsigned i) {
                    auto &group = cell_group[i];

                    util::profiler_enter("stepping","events");
                    group.enqueue_events(communicator.queue(i));
                    util::profiler_leave();

                    group.advance(tstep, dt);

                    util::profiler_enter("events");
                    communicator.add_spikes(group.spikes());
                    group.clear_spikes();
                    util::profiler_leave(2);
                });

            util::profiler_enter("stepping", "exchange");
            communicator.exchange();
            util::profiler_leave(2);

            t_ += delta;
        }
        return t_;
    }

    explicit model(const recipe &rec, float sample_dt) {
        // crude load balancing:
        auto num_domains = global_policy::size();
        auto domain_id = global_policy::id();
        auto num_cells = rec.num_cells();

        cell_gid_type cell_from = (cell_gid_type)(num_cells*(domain_id/(double)num_domains));
        cell_gid_type cell_to = (cell_gid_type)(num_cells*((domain_id+1)/(double)num_domains));

        // construct cell groups (one cell per group) and attach samplers
        cell_groups.resize(cell_to-cell_from);
        samplers.resize(cell_to-cell_from);

        threading::parallel_for::apply(cell_from, cell_to, 
            [&](cell_gid_type i) {
                util::profiler_enter("setup", "cells");
                auto cell = cell_group(rec.get_cell(i));
                auto idx = i-cell_from;
                cell_groups[idx] = cell_group(cell);

                cell_local_index_type j = 0;
                for (const auto& probe: cell.probes()) {
                    samplers[idx].emplace_back({i,j}, probe.kind, probe.location, sample_dt);
                    const auto &sampler = samplers[idx].back();
                    cell_groups[idx].add_sampler(sampler, sampler.next_sample_t());
                }
                util::profiler_leave(2);
            });

        // initialise communicator
        communicator = communicator_type(cell_from, cell_to);
    }

private:
    double t_ = 0.;
    std::vector<cell_group> cell_groups;
    std::vector<std::vector<sample_to_trace>> samplers;
    communicator_type communicator;
};


// move sampler code to another source file...
struct sample_trace {
    struct sample_type {
        float time;
        double value;
    };

    std::string name;
    std::string units;
    cell_gid_type cell_gid;
    cell_index_type probe_index;
    std::vector<sample_type> samples;
};

struct sample_to_trace {
    float next_sample_t() const { return t_next_sample_; }

    optional<float> operator()(float t, double v) {
        if (t<t_next_sample_) {
            return t_next_sample_;
        }

        trace.samples.push_back({t,v});
        return t_next_sample_+=sample_dt_;
    }

    sample_to_trace(cell_member_type probe_id,
                    const std::string &name,
                    const std::string &units,
                    float dt,
                    float t_start=0):
        trace_{{name, units, probe_id.gid, probe_id.index}},
        sample_dt_(dt),
        t_next_sample_(t_start)
    {}

    sample_to_trace(cell_member_type probe_id,
                    probeKind kind,
                    segment_location loc,
                    float dt,
                    float t_start=0):
        sample_to_trace(probe_id, "", "", dt, t_start)
    {
        std::string name = "";
        std::string units = "";

        switch (kind) {
        case probeKind::mebrane_voltage:
            name = "v";
            units = "mV";
            break;
        case probeKind::mebrane_current:
            name = "i";
            units = "mA/cm^2";
            break;
        default: ;
        }

        trace_.name = name + (loc.segment? "dend": "soma");
        trace_.units = units;
    }

    void write_trace(const std::string& prefix = "trace_") const {
        // do not call during simulation: thread-unsafe access to traces.
        auto path = prefix + std::to_string(trace.cell_gid) +
                    "." + std::to_string(trace.probe_index) + ".json";

        nlohmann::json jrep;
        jrep["name"] = trace.name;
        jrep["units"] = trace.units;
        jrep["cell"] = trace.cell_gid;
        jrep["probe"] = trace.probe_index;

        auto& jt = jrep["data"]["time"];
        auto& jy = jrep["data"][trace.name];

        for (const auto& sample: trace.samples) {
            jt.push_back(sample.time);
            jy.push_back(sample.value);
        }
        std::ofstream file(path);
        file << std::setw(1) << jrep << std::endl;
    }

private:
    sample_trace trace_;

    float sample_dt_;
    float t_next_sample_;

};

} // namespace mc
} // namespace nest
