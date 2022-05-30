#include <arbor/common_types.hpp>

#include "backends/gpu/event_stream.hpp"
#include "memory/memory.hpp"

namespace arb {
namespace gpu {

void event_stream_base::clear() {
    span_begin_ = span_end_ = 0;
}

void event_stream_base::mark_until_after(const fvm_value_type& t_until) {
    using ::arb::event_time;

    const index_type end = host_ev_time_.size();
    while (span_end_!=end && !(host_ev_time_[span_end_]>t_until)) {
        ++span_end_;
    }
}

void event_stream_base::mark_until(const fvm_value_type& t_until) {
    using ::arb::event_time;

    const index_type end = host_ev_time_.size();
    while (span_end_!=end && t_until>host_ev_time_[span_end_]) {
        ++span_end_;
    }
}

} // namespace gpu
} // namespace arb
