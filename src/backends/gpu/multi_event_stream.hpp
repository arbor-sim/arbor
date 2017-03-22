#pragma once

// Indexed collection of pop-only event queues --- multicore back-end implementation.

#include <common_types.hpp>
#include <backends/event.hpp>
#include <memory/array.hpp>
#include <memory/copy.hpp>

namespace nest {
namespace mc {
namespace gpu {

class multi_event_stream {
public:
    using size_type = cell_size_type;
    using value_type = double;

    using array = memory::device_vector<value_type>;
    using iarray = memory::device_vector<size_type>;

    using const_view = array::const_view_type;
    using view = array::view_type;

    multi_event_stream() {}

    explicit multi_event_stream(size_type n_stream):
        n_stream_(n_stream),
        span_begin_(n_stream),
        span_end_(n_stream),
        mark_(n_stream),
        n_nonempty_stream_(1)
    {}

    size_type n_streams() const { return n_stream_; }

    bool empty() const { return n_nonempty_stream_[0]==0; }

    void clear();

    // Initialize event streams from a vector of `deliverable_event`.
    void init(const std::vector<deliverable_event>& staged);

    // Designate for processing events `ev` at head of each event stream `i`
    // until `event_time(ev)` > `t_until[i]`.
    void mark_until_after(const_view t_until);

    // Remove marked events from front of each event stream.
    void drop_marked_events();

    // If the head of `i`th event stream exists and has time less than `t_until[i]`, set
    // `t_until[i]` to the event time.
    void event_time_if_before(view t_until);

    // Interface for access by mechanism kernels:
    struct span_state {
        size_type n;
        const size_type* ev_mech_id;
        const size_type* ev_index;
        const value_type* ev_weight;
        const size_type* span_begin;
        const size_type* mark;
    };

    span_state delivery_data() const {
        return {n_stream_, ev_mech_id_.data(), ev_index_.data(), ev_weight_.data(), span_begin_.data(), mark_.data()};
    }

private:
    size_type n_stream_;
    array ev_time_;
    array ev_weight_;
    iarray ev_mech_id_;
    iarray ev_index_;
    iarray span_begin_;
    iarray span_end_;
    iarray mark_;
    iarray n_nonempty_stream_;
};

} // namespace gpu
} // namespace nest
} // namespace mc
