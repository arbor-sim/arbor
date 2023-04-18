#pragma once

// Indexed collection of pop-only event queues --- multicore back-end implementation.

#include "backends/event_stream_base.hpp"
#include "util/range.hpp"
#include "util/rangeutil.hpp"

namespace arb {
namespace multicore {

template <typename Event>
class event_stream : public event_stream_base<Event, util::range<::arb::event_data_type<Event>*>> {
public:
    using base = event_stream_base<Event, util::range<::arb::event_data_type<Event>*>>;
    using size_type = typename base::size_type;

    event_stream() = default;

    using base::clear;

    // Initialize event streams from a vector of vector of events
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
        base::ev_spans_.reserve(staged.size());
        base::ev_data_.reserve(num_events);

        // add event data and spans
        for (const auto& v : staged) {
            auto ptr = base::ev_data_.data() + base::ev_data_.size();
            base::ev_spans_.emplace_back(ptr, ptr + v.size());
            for (const auto& ev : v) {
                base::ev_data_.push_back(event_data(ev));
            }
        }

        arb_assert(num_events == base::ev_data_.size());
    }

    ARB_SERDES_ENABLE(event_stream<Event>, ev_data_, ev_spans_);
};

} // namespace multicore
} // namespace arb
