#pragma once

#include <type_traits>
#include <vector>

#include <arbor/generic_event.hpp>

#include "backends/event.hpp"
#include "backends/event_stream_state.hpp"

namespace arb {

template <typename Event, typename Span>
class event_stream_base {
public: // member types
    using size_type = std::size_t;
    using event_type = Event;
    using event_time_type = ::arb::event_time_type<Event>;
    using event_data_type = ::arb::event_data_type<Event>;

protected: // private member types
    using span_type = Span;

    static_assert(std::is_same<decltype(std::declval<span_type>().begin()), event_data_type*>::value);
    static_assert(std::is_same<decltype(std::declval<span_type>().end()), event_data_type*>::value);

protected: // members
    std::vector<event_data_type> ev_data_;
    std::vector<span_type> ev_spans_;
    size_type index_ = 0;

public:
    event_stream_base() = default;

    // returns true if the currently marked time step has no events
    bool empty() const {
        return ev_spans_.empty() || ev_data_.empty() || !index_ || index_ > ev_spans_.size() ||
            !ev_spans_[index_-1].size();
    }

    void mark() {
        index_ += (index_ <= ev_spans_.size() ? 1 : 0);
    }

    auto marked_events() {
        using std::begin;
        using std::end;
        if (empty()) {
            return make_event_stream_state((event_data_type*)nullptr, (event_data_type*)nullptr);
        } else {
            return make_event_stream_state(begin(ev_spans_[index_-1]), end(ev_spans_[index_-1]));
        }
    }

    // clear all previous data
    void clear() {
        ev_data_.clear();
        ev_spans_.clear();
        index_ = 0;
    }

};

} // namespace arb
