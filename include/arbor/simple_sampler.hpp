#pragma once

/*
 * Simple(st?) implementation of a recorder of scalar
 * trace data from a cell probe, with some metadata.
 */

#include <stdexcept>
#include <type_traits>
#include <vector>

#include <arbor/common_types.hpp>
#include <arbor/sampling.hpp>
#include <arbor/util/any_ptr.hpp>

namespace arb {

template <typename V>
struct trace_entry {
    time_type t;
    V v;
};

template <typename V>
using trace_data = std::vector<trace_entry<V>>;

template <typename V, typename = std::enable_if_t<std::is_trivially_copyable<V>::value>>
class simple_sampler {
public:
    explicit simple_sampler(trace_data<V>& trace): trace_(trace) {}

    void operator()(cell_member_type probe_id, probe_tag tag, std::size_t n, const sample_record* recs) {
        for (std::size_t i = 0; i<n; ++i) {
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
