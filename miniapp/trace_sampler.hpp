#pragma once

#include <cstdlib>
#include <vector>

#include <catypes.hpp>
#include <cell.hpp>
#include <util/optional.hpp>

namespace nest {
namespace mc {

// move sampler code to another source file...
struct sample_trace {
    struct sample_type {
        float time;
        double value;
    };

    std::string name;
    std::string units;
    cell_gid_type cell_gid;
    cell_local_index_type probe_index;
    std::vector<sample_type> samples;
};

struct sample_to_trace {
    float next_sample_t() const { return t_next_sample_; }

    util::optional<float> operator()(float t, double v) {
        if (t<t_next_sample_) {
            return t_next_sample_;
        }

        trace_.samples.push_back({t,v});
        return t_next_sample_+=sample_dt_;
    }

    sample_to_trace(cell_member_type probe_id,
                    const std::string &name,
                    const std::string &units,
                    float dt,
                    float t_start=0);

    sample_to_trace(cell_member_type probe_id,
                    probeKind kind,
                    segment_location loc,
                    float dt,
                    float t_start=0);

    void write_trace(const std::string& prefix = "trace_") const;

    const sample_trace& trace() const { return trace_; }

private:
    sample_trace trace_;

    float sample_dt_;
    float t_next_sample_;

};

} // namespace mc
} // namespace nest
