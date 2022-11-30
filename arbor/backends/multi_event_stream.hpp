#pragma once

// Indexed collection of pop-only event queues

#include <limits>
#include <ostream>
#include <utility>
#include <vector>

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

    multi_event_stream() {}

    size_type n_streams() const { return span_begin_.size(); }

    size_type n_remaining() const { return remaining_; }

    size_type n_marked() const { return marked_; }

    bool empty() const { return remaining_==0u; }

    void clear() {
        ev_time_.clear();
        ev_index_.clear();
        ev_data_.clear();
        span_begin_.clear();
        span_end_.clear();
        offsets_.clear();
        remaining_ = 0;
        marked_ = 0;
    }

    // Initialize event streams from a vector of events, sorted by index and then by time.
    void init(const std::vector<Event>& staged) {
        using ::arb::event_time;
        using ::arb::event_index;
        using ::arb::event_data;

        if (staged.size()>std::numeric_limits<size_type>::max()) {
            throw arbor_internal_error("multi_event_stream: too many events for size type");
        }

        // clear all earlier data
        clear();

        // Staged events should already be sorted by index.
        arb_assert(util::is_sorted_by(staged, [](const Event& ev) { return event_index(ev); }));

        util::assign_by(ev_data_, staged, [](const Event& ev) { return event_data(ev); });
        util::assign_by(ev_time_, staged, [](const Event& ev) { return event_time(ev); });

        size_type i=0;
        for (const Event& ev : staged) {
            if (auto idx = event_index(ev); ev_index_.size() == 0u || ev_index_.back() < idx) {
                ev_index_.push_back(idx);
                offsets_.push_back(i);
            }
            ++i;
        }
        span_begin_.assign(offsets_.begin(), offsets_.end());
        span_end_.assign(offsets_.begin(), offsets_.end());
        offsets_.push_back(i);

        remaining_ = i;
        arb_assert(remaining_ == staged.size());

        // Within a subrange of events with the same index, events should be sorted by time.
        for (size_type k=0; k<n_streams(); ++k) {
            arb_assert(std::is_sorted(ev_time_.begin()+offsets_[k], ev_time_.begin()+offsets_[k+1]));
        }
    }

    // Designate for processing events `ev` at head of each event stream
    // until `event_time(ev)` > `t_until`.
    void mark_until_after(arb_value_type t_until) {
        for (size_type i=0; i<n_streams(); ++i) {
            const size_type end = offsets_[i+1];
            while (span_end_[i]!=end && ev_time_[span_end_[i]]<=t_until) {
                ++span_end_[i];
            }
            marked_ += span_end_[i] - span_begin_[i];
        }
    }

    // Remove marked events from front of each event stream.
    void drop_marked_events() {
        for (size_type i=0; i<n_streams(); ++i) {
            span_begin_[i] = span_end_[i];
        }
        remaining_ -= marked_;
        marked_ = 0;
    }

    // Interface for access to marked events by mechanisms/kernels:
    state marked_events() const {
        return {n_streams(), n_marked(), ev_data_.data(), span_begin_.data(), span_end_.data()};
    }

    friend std::ostream& operator<<(std::ostream& out, const multi_event_stream<Event>& m) {
        out << "[";
        for (arb_size_type k=0; k<m.n_streams(); ++k) {
            out << " { " << m.ev_index_[k] << " : ";
            const auto a = m.offsets_[k];
            const auto b = m.span_begin_[k];
            const auto c = m.span_end_[k];
            const auto d = m.offsets_[k+1];
            for (arb_size_type i=a; i<d; ++i) {
                const bool discarded = i<b;
                const bool marked = (i>=b && i<c);
                if (discarded) {
                    out << "        x";
                }
                else {
                    out << util::strprintf(" % 7.3f%c", m.ev_time_[i], marked?'*':' ');
                }
            }
            out << " },";
        }
        return out << "]";
    }

protected:
    std::vector<event_time_type> ev_time_;
    std::vector<event_index_type> ev_index_;
    std::vector<event_data_type> ev_data_;
    std::vector<size_type> span_begin_;
    std::vector<size_type> span_end_;
    std::vector<size_type> offsets_;
    size_type remaining_ = 0;
    size_type marked_ = 0;
};

} // namespace arb
