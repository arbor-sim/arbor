#include <arbor/common_types.hpp>

#include "backends/gpu/multi_event_stream.hpp"
#include "memory/memory.hpp"

namespace arb {
namespace gpu {

// These wrappers are implemented in the multi_event_stream.cu file, which
// is spearately compiled by nvcc, to protect nvcc from having to parse C++17.
void mark_until_after_w(unsigned n,
        arb_index_type* mark,
        arb_index_type* span_end,
        arb_value_type* ev_time,
        const arb_value_type* t_until);
void mark_until_w(unsigned n,
        arb_index_type* mark,
        arb_index_type* span_end,
        arb_value_type* ev_time,
        const arb_value_type* t_until);
void drop_marked_events_w(unsigned n,
        arb_index_type* n_nonempty_stream,
        arb_index_type* span_begin,
        arb_index_type* span_end,
        arb_index_type* mark);
void event_time_if_before_w(unsigned n,
        arb_index_type* span_begin,
        arb_index_type* span_end,
        arb_value_type* ev_time,
        arb_value_type* t_until);

void multi_event_stream_base::clear() {
    memory::fill(span_begin_, 0u);
    memory::fill(span_end_, 0u);
    memory::fill(mark_, 0u);
    n_nonempty_stream_[0] = 0;
}

// Designate for processing events `ev` at head of each event stream `i`
// until `event_time(ev)` > `t_until[i]`.
void multi_event_stream_base::mark_until_after(const_view t_until) {
    arb_assert(n_streams()==t_until.size());
    mark_until_after_w(n_stream_, mark_.data(), span_end_.data(), ev_time_.data(), t_until.data());
}

// Designate for processing events `ev` at head of each event stream `i`
// while `t_until[i]` > `event_time(ev)`.
void multi_event_stream_base::mark_until(const_view t_until) {
    mark_until_w(n_stream_, mark_.data(), span_end_.data(), ev_time_.data(), t_until.data());
}

// Remove marked events from front of each event stream.
void multi_event_stream_base::drop_marked_events() {
    drop_marked_events_w(n_stream_, n_nonempty_stream_.data(), span_begin_.data(), span_end_.data(), mark_.data());
}

// If the head of `i`th event stream exists and has time less than `t_until[i]`, set
// `t_until[i]` to the event time.
void multi_event_stream_base::event_time_if_before(view t_until) {
    event_time_if_before_w(n_stream_, span_begin_.data(), span_end_.data(), ev_time_.data(), t_until.data());
}

} // namespace gpu
} // namespace arb
