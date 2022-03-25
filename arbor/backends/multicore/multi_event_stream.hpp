#pragma once

// Indexed collection of pop-only event queues --- multicore back-end implementation.

#include <limits>
#include <ostream>
#include <utility>

#include <arbor/assert.hpp>
#include <arbor/arbexcept.hpp>
#include <arbor/fvm_types.hpp>
#include <arbor/generic_event.hpp>

#include "backends/event.hpp"
#include "backends/multi_event_stream_state.hpp"
#include "util/range.hpp"
#include "util/rangeutil.hpp"
#include "util/strprintf.hpp"

namespace arb {
namespace multicore {

template <typename Event>
class multi_event_stream {
public:
    using size_type = fvm_size_type;
    using index_type = fvm_index_type;
    using event_type = Event;

    using event_time_type = ::arb::event_time_type<Event>;
    using event_data_type = ::arb::event_data_type<Event>;

    using state = multi_event_stream_state<event_data_type>;

    multi_event_stream() {}

    bool empty() const { return span_begin_==ev_data_.size(); }

    void clear() {
        ev_data_.clear();

        span_begin_ = 0;
        span_end_ = 0;
    }

    // Initialize event streams from a vector of events, sorted by time.
    void init(std::vector<Event> staged) {
        using ::arb::event_time;
        using ::arb::event_data;

        if (staged.size()>std::numeric_limits<size_type>::max()) {
            throw arbor_internal_error("multicore/multi_event_stream: too many events for size type");
        }

        util::assign_by(ev_data_, staged, [](const Event& ev) { return event_data(ev); });
        util::assign_by(ev_time_, staged, [](const Event& ev) { return event_time(ev); });

        span_begin_ = span_end_ = 0;
    }

    // Designate for processing events `ev` at head of each event stream `i`
    // until `event_time(ev)` > `t_until[i]`.
    void mark_until_after(const fvm_value_type& t_until) {
        using ::arb::event_time;

        const index_type end = ev_time_.size();
        while (span_end_!=end && !(ev_time_[span_end_]>t_until)) {
            ++span_end_;
        }
    }

    // Designate for processing events `ev` at head of each event stream `i`
    // while `t_until[i]` > `event_time(ev)`.
    void mark_until(const arb::fvm_value_type& t_until) {
        using ::arb::event_time;

        const index_type end = ev_time_.size();
        while (span_end_!=end && t_until>ev_time_[span_end_]) {
            ++span_end_;
        }
    }

    // Remove marked events from front of each event stream.
    void drop_marked_events() {
        span_begin_ = span_end_;
    }

    // Interface for access to marked events by mechanisms/kernels:
    state marked_events() const {
        return {1, ev_data_.data(), &span_begin_, &span_end_};
    }

    friend std::ostream& operator<<(std::ostream& out, const multi_event_stream<Event>& m) {
        /*
        auto n_ev = m.ev_data_.size();
        auto n = m.n_streams();

        out << "\n[";
        unsigned i = 0;
        for (unsigned ev_i = 0; ev_i<n_ev; ++ev_i) {
            while (i<n && m.span_end_[i]<=ev_i) ++i;
            out << (i<n? util::strprintf(" % 7d ", i): "      ?");
        }
        out << "]\n[";

        i = 0;
        for (unsigned ev_i = 0; ev_i<n_ev; ++ev_i) {
            while (i<n && m.span_end_[i]<=ev_i) ++i;

            bool discarded = i<n && m.span_begin_[i]>ev_i;
            bool marked = i<n && m.mark_[i]>ev_i;

            if (discarded) {
                out << "        x";
            }
            else {
                out << util::strprintf(" % 7.3f%c", m.ev_time_[ev_i], marked?'*':' ');
            }
        }
        out << "]\n";
        */
        return out << "[multi_event_stream thingy]";
    }

private:
    std::vector<event_time_type> ev_time_;
    std::vector<event_data_type> ev_data_;
    index_type span_begin_ = 0;
    index_type span_end_ = 0;
    size_type remaining() const {
        return ev_time_.size() - span_begin_;
    };
};

} // namespace multicore
} // namespace arb
