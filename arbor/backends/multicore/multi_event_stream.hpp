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
    using size_type = arb_size_type;
    using index_type = arb_index_type;
    using event_type = Event;

    using event_time_type = ::arb::event_time_type<Event>;
    using event_data_type = ::arb::event_data_type<Event>;
    using event_index_type = ::arb::event_index_type<Event>;

    using state = multi_event_stream_state<event_data_type>;

    multi_event_stream() = default;

    explicit multi_event_stream(size_type n_stream):
       span_begin_(n_stream), span_end_(n_stream), mark_(n_stream) {}

    size_type n_streams() const { return span_begin_.size(); }

    bool empty() const { return remaining_==0; }

    void clear() {
        ev_data_.clear();
        remaining_ = 0;

        util::fill(span_begin_, 0);
        util::fill(span_end_, 0);
        util::fill(mark_, 0);
    }

    // Initialize event streams from a vector of events, sorted by time.
    void init(std::vector<Event> staged) {
        using ::arb::event_time;
        using ::arb::event_index;
        using ::arb::event_data;

        if (staged.size()>std::numeric_limits<size_type>::max()) {
            throw arbor_internal_error("multicore/multi_event_stream: too many events for size type");
        }

        // Staged events should already be sorted by index.
        arb_assert(util::is_sorted_by(staged, [](const Event& ev) { return event_index(ev); }));

        std::size_t n_ev = staged.size();
        util::assign_by(ev_data_, staged, [](const Event& ev) { return event_data(ev); });
        util::assign_by(ev_time_, staged, [](const Event& ev) { return event_time(ev); });

        // Determine divisions by `event_index` in ev list.
        arb_assert(n_streams() == span_begin_.size());
        arb_assert(n_streams() == span_end_.size());
        arb_assert(n_streams() == mark_.size());

        index_type ev_begin_i = 0;
        index_type ev_i = 0;
        for (size_type s = 0; s<n_streams(); ++s) {
            while ((size_type)ev_i<n_ev && (size_type)(event_index(staged[ev_i]))<s+1) ++ev_i;

            // Within a subrange of events with the same index, events should
            // be sorted by time.
            arb_assert(util::is_sorted(util::subrange_view(ev_time_, ev_begin_i, ev_i)));
            mark_[s] = ev_begin_i;
            span_begin_[s] = ev_begin_i;
            span_end_[s] = ev_i;
            ev_begin_i = ev_i;
        }

        remaining_ = n_ev;
    }

    // Designate for processing events `ev` at head of each event stream `i`
    // until `event_time(ev)` > `t_until[i]`.
    template <typename TimeSeq>
    void mark_until_after(const TimeSeq& t_until) {
        using ::arb::event_time;

        arb_assert(n_streams()==std::size(t_until));

        // note: operation on each `i` is independent.
        for (size_type i = 0; i<n_streams(); ++i) {
            auto end = span_end_[i];
            auto t = t_until[i];

            auto mark = span_begin_[i];
            while (mark!=end && !(ev_time_[mark]>t)) {
                ++mark;
            }
            mark_[i] = mark;
        }
    }

    // Designate for processing events `ev` at head of each event stream `i`
    // while `t_until[i]` > `event_time(ev)`.
    template <typename TimeSeq>
    void mark_until(const TimeSeq& t_until) {
        using ::arb::event_time;

        arb_assert(n_streams()==std::size(t_until));

        // note: operation on each `i` is independent.
        for (size_type i = 0; i<n_streams(); ++i) {
            auto end = span_end_[i];
            auto t = t_until[i];

            auto mark = span_begin_[i];
            while (mark!=end && t>ev_time_[mark]) {
                ++mark;
            }
            mark_[i] = mark;
        }
    }

    // Remove marked events from front of each event stream.
    void drop_marked_events() {
        // note: operation on each `i` is independent.
        for (size_type i = 0; i<n_streams(); ++i) {
            remaining_ -= (mark_[i]-span_begin_[i]);
            span_begin_[i] = mark_[i];
        }
    }

    // Interface for access to marked events by mechanisms/kernels:
    state marked_events() const {
        return {n_streams(), ev_data_.data(), span_begin_.data(), mark_.data()};
    }

    // If the head of `i`th event stream exists and has time less than `t_until[i]`, set
    // `t_until[i]` to the event time.
    template <typename TimeSeq>
    void event_time_if_before(TimeSeq& t_until) {
        using ::arb::event_time;

        // note: operation on each `i` is independent.
        for (size_type i = 0; i<n_streams(); ++i) {
            if (span_begin_[i]==span_end_[i]) {
               continue;
            }

            auto ev_t = ev_time_[span_begin_[i]];
            if (t_until[i]>ev_t) {
                t_until[i] = ev_t;
            }
        }
    }

    friend std::ostream& operator<<(std::ostream& out, const multi_event_stream<Event>& m) {
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
        return out;
    }

private:
    std::vector<event_time_type> ev_time_;
    std::vector<index_type> span_begin_;
    std::vector<index_type> span_end_;
    std::vector<index_type> mark_;
    std::vector<event_data_type> ev_data_;
    size_type remaining_ = 0;
};

} // namespace multicore
} // namespace arb
