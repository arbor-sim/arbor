#pragma once

#include <vector>

#include <arbor/generic_event.hpp>

#include "backends/event.hpp"
#include "backends/event_stream_state.hpp"

namespace arb {

template <typename Event>
class event_stream_base {
public: // member types
    using size_type = std::size_t;
    using event_type = Event;
    using event_time_type = ::arb::event_time_type<Event>;
    using event_data_type = ::arb::event_data_type<Event>;

protected: // private member types

protected: // members
    std::vector<event_data_type> ev_data_;
    std::vector<std::size_t> ev_spans_;
    size_type index_ = 0;

public:
    event_stream_base() = default;

    // returns true if the currently marked time step has no events
    bool empty() const {
        return ev_spans_.empty()
            || ev_data_.empty()
            || !index_
            || index_ > ev_spans_.size()
            || ev_spans_[index_-1] >= ev_spans_[index_];
    }

    void mark() {
        index_ += 1;
    }

    auto marked_events() {
        if (empty()) {
            return make_event_stream_state((event_data_type*)nullptr, (event_data_type*)nullptr);
        } else {
            auto ptr = ev_data_.data();
            return make_event_stream_state(ptr + ev_spans_[index_-1],
                                           ptr + ev_spans_[index_]);
        }
    }

    // clear all previous data
    void clear() {
        ev_data_.clear();
        ev_spans_.clear();
        ev_spans_.push_back(0);
        index_ = 0;
    }
};

} // namespace arb
