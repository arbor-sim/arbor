#pragma once

/*
 * Simple(st?) implementation of a recorder of scalar
 * trace data from a cell probe, with some metadata.
 */

#include <vector>

#include <common_types.hpp>
#include <sampling.hpp>
#include <util/any_ptr.hpp>
#include <util/deduce_return.hpp>
#include <util/span.hpp>
#include <util/transform.hpp>

#include <iostream>

namespace arb {

template <typename V>
struct trace_entry {
    time_type t;
    V v;
};

template <typename V>
using trace_data = std::vector<trace_entry<V>>;

// NB: work-around for lack of function return type deduction
// in C++11; can't use lambda within DEDUCED_RETURN_TYPE.

namespace impl {
    template <typename V>
    inline float time(const trace_entry<V>& x) { return x.t; }

    template <typename V>
    inline const V& value(const trace_entry<V>& x) { return x.v; }
}

template <typename V>
inline auto times(const trace_data<V>& trace) DEDUCED_RETURN_TYPE(
   util::transform_view(trace, impl::time<V>)
)

template <typename V>
inline auto values(const trace_data<V>& trace) DEDUCED_RETURN_TYPE(
   util::transform_view(trace, impl::value<V>)
)

template <typename V, typename = util::enable_if_trivially_copyable_t<V>>
class simple_sampler {
public:
    explicit simple_sampler(trace_data<V>& trace): trace_(trace) {}

    void operator()(cell_member_type probe_id, probe_tag tag, std::size_t n, const sample_record* recs) {
        for (auto i: util::make_span(0, n)) {
            if (auto p = util::any_cast<const V*>(recs[i].data)) {
                trace_.push_back({recs[i].time, *p});
            }
            else {
                throw std::runtime_error("unexpected sample type in simple_sampler");
            }
        }
    }

private:
    trace_data<V>& trace_;
};

template <typename V>
inline simple_sampler<V> make_simple_sampler(trace_data<V>& trace) {
    return simple_sampler<V>(trace);
}

} // namespace arb
