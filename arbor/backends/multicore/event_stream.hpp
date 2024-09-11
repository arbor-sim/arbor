#pragma once

// Indexed collection of pop-only event queues --- multicore back-end implementation.

#include "arbor/spike_event.hpp"
#include "backends/event_stream_base.hpp"
#include "timestep_range.hpp"

namespace arb {
namespace multicore {

template <typename Event>
struct event_stream: public event_stream_base<Event> {
    using base = event_stream_base<Event>;
    using size_type = typename base::size_type;

    using base::clear;
    using base::ev_spans_;
    using base::ev_data_;
    using base::base_ptr_;

    event_stream() = default;

    // Initialize event stream from a vector of vector of events
    // Outer vector represents time step bins
    void init(const std::vector<std::vector<Event>>& staged) {
        // clear previous data
        clear();

        // return if there are no timestep bins
        if (!staged.size()) return;

        // return if there are no events
        const size_type num_events = util::sum_by(staged, [] (const auto& v) {return v.size();});
        if (!num_events) return;

        // allocate space for spans and data
        ev_spans_.reserve(staged.size() + 1);
        ev_data_.reserve(num_events);

        // add event data and spans
        for (const auto& v : staged) {
            for (const auto& ev: v) ev_data_.push_back(event_data(ev));
            ev_spans_.push_back(ev_data_.size());
        }

        arb_assert(num_events == ev_data_.size());
        arb_assert(staged.size() + 1 == ev_spans_.size());
        base_ptr_ = ev_data_.data();
    }

    // Initialize event stream assuming ev_data_ and ev_span_ has
    // been set previously (e.g. by `base::multi_event_stream`)
    void init() { base_ptr_ = ev_data_.data(); }

    ARB_SERDES_ENABLE(event_stream<Event>,
                      ev_data_,
                      ev_spans_,
                      index_);
};
} // namespace multicore
} // namespace arb
