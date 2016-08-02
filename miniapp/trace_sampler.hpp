#pragma once

#include <cstdlib>
#include <vector>

#include <catypes.hpp>
#include <cell.hpp>
#include <util/optional.hpp>

#include <iostream>

namespace nest {
namespace mc {

// move sampler code to another source file...
template <typename TimeT=float, typename ValueT=double>
struct sample_trace {
    using time_type = TimeT;
    using value_type = ValueT;

    struct sample_type {
        time_type time;
        value_type value;
    };

    std::string name;
    std::string units;
    cell_member_type probe_id;
    std::vector<sample_type> samples;

    sample_trace() =default;
    sample_trace(cell_member_type probe_id, const std::string &name, const std::string &units):
        name(name), units(units), probe_id(probe_id)
    {}
};

template <typename TimeT=float, typename ValueT=double>
struct trace_sampler {
    using time_type = TimeT;
    using value_type = ValueT;

    float next_sample_t() const { return t_next_sample_; }

    util::optional<time_type> operator()(time_type t, value_type v) {
        if (t<t_next_sample_) {
            return t_next_sample_;
        }

        trace_->samples.push_back({t,v});
        return t_next_sample_+=sample_dt_;
    }

    trace_sampler(sample_trace<time_type, value_type> *trace, time_type sample_dt, time_type tfrom=0):
       trace_(trace), sample_dt_(sample_dt), t_next_sample_(tfrom)
    {}

private:
    sample_trace<time_type, value_type> *trace_;

    time_type sample_dt_;
    time_type t_next_sample_;
};

// with type deduction ...
template <typename TimeT, typename ValueT>
trace_sampler<TimeT, ValueT> make_trace_sampler(sample_trace<TimeT, ValueT> *trace, TimeT sample_dt, TimeT tfrom=0) {
    return trace_sampler<TimeT, ValueT>(trace, sample_dt, tfrom);
}

} // namespace mc
} // namespace nest
