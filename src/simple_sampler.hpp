#pragma once

/*
 * Simple(st?) implementation of a recorder of scalar
 * trace data from a cell probe, with some metadata.
 */

#include <functional>
#include <vector>

#include <common_types.hpp>
#include <sampler_function.hpp>
#include <util/optional.hpp>
#include <util/deduce_return.hpp>
#include <util/transform.hpp>

namespace nest {
namespace mc {

struct trace_entry {
    float t;
    double v;
};

using trace_data = std::vector<trace_entry>;

// NB: work-around for lack of function return type deduction
// in C++11; can't use lambda within DEDUCED_RETURN_TYPE.

namespace impl {
    inline float time(const trace_entry& x) { return x.t; }
    inline float value(const trace_entry& x) { return x.v; }
}

inline auto times(const trace_data& trace) DEDUCED_RETURN_TYPE(
   util::transform_view(trace, impl::time)
)

inline auto values(const trace_data& trace) DEDUCED_RETURN_TYPE(
   util::transform_view(trace, impl::value)
)

class simple_sampler {
public:
    trace_data trace;

    simple_sampler(time_type dt, time_type t0=0):
        t0_(t0),
        sample_dt_(dt),
        t_next_sample_(t0)
    {}

    void reset() {
        trace.clear();
        t_next_sample_ = t0_;
    }

    sampler_function sampler() {
        return [&](time_type t, double v) -> util::optional<time_type> {
            if (t<t_next_sample_) {
                return t_next_sample_;
            }
            else {
                trace.push_back({t, v});
                return t_next_sample_+=sample_dt_;
            }
        };
    }

private:
    time_type t0_ = 0;
    time_type sample_dt_ = 0;
    time_type t_next_sample_ = 0;
};

} // namespace mc
} // namespace nest
