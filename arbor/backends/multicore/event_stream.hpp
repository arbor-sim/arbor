#pragma once

// Indexed collection of pop-only event queues --- multicore back-end implementation.

#include <iosfwd>
#include <limits>
#include <utility>

#include <arbor/arbexcept.hpp>
#include <arbor/fvm_types.hpp>
#include <arbor/generic_event.hpp>

#include "backends/event.hpp"
#include "backends/event_stream_state.hpp"
#include "util/range.hpp"
#include "util/rangeutil.hpp"
#include "util/strprintf.hpp"

namespace arb {
namespace multicore {

template <typename Event>
class event_stream {
public:
    using size_type = arb_size_type;
    using event_type = Event;

    using event_time_type = ::arb::event_time_type<Event>;
    using event_data_type = ::arb::event_data_type<Event>;

    using state = event_stream_state<event_data_type>;

    event_stream() {}

    bool empty() const { return span_begin_==ev_data_.size(); }

    void clear() {
        ev_time_.clear();
        ev_data_.clear();
        span_begin_ = 0;
        span_end_ = 0;
    }

    // Initialize event streams from a vector of events, sorted by time.
    void init(const std::vector<Event>& staged) {
        using ::arb::event_time;
        using ::arb::event_data;

        if (staged.size()>std::numeric_limits<size_type>::max()) {
            throw arbor_internal_error("multicore/event_stream: too many events for size type");
        }

        clear();
        util::assign_by(ev_data_, staged, [](const Event& ev) { return event_data(ev); });
        util::assign_by(ev_time_, staged, [](const Event& ev) { return event_time(ev); });
    }

    // Designate for processing events `ev` at head of the event stream
    // until `event_time(ev)` > `t_until`.
    void mark_until_after(arb_value_type t_until) {
        using ::arb::event_time;
        const auto end = ev_time_.size();
        while (span_end_!=end && ev_time_[span_end_]<=t_until) {
            ++span_end_;
        }
    }

    // Designate for processing events `ev` at head the stream
    // while `t_until` > `event_time(ev)`.
    void mark_until(arb_value_type t_until) {
        using ::arb::event_time;
        const auto end = ev_time_.size();
        while (span_end_!=end && ev_time_[span_end_]<t_until) {
            ++span_end_;
        }
    }

    // Remove marked events from front of the event stream.
    void drop_marked_events() {
        span_begin_ = span_end_;
    }

    // Interface for access to marked events by mechanisms/kernels:
    state marked_events() const {
        return {ev_data_.data()+span_begin_, ev_data_.data()+span_end_};
    }

    template<class CharT, class Traits = std::char_traits<CharT>>
    friend std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& out, const event_stream<Event>& m) {
        const auto n_ev = m.ev_data_.size();

        out << "[";

        for (size_type ev_i = 0; ev_i<n_ev; ++ev_i) {
            bool discarded = ev_i<m.span_begin_;
            bool marked = !discarded && ev_i<m.span_end_;

            if (discarded) {
                out << "        x";
            }
            else {
                out << util::strprintf(" % 7.3f%c", m.ev_time_[ev_i], marked?'*':' ');
            }
        }
        return out << "]";
    }

protected:
    std::vector<event_time_type> ev_time_;
    std::vector<event_data_type> ev_data_;

    size_type span_begin_ = 0;
    size_type span_end_ = 0;
};

} // namespace multicore
} // namespace arb
