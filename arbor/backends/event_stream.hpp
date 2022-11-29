#pragma once

#include <limits>
#include <ostream>
#include <utility>
#include <vector>

#include <arbor/assert.hpp>
#include <arbor/arbexcept.hpp>
#include <arbor/fvm_types.hpp>
#include <arbor/generic_event.hpp>

#include "backends/event.hpp"
#include "backends/event_stream_state.hpp"
#include "util/range.hpp"
#include "util/rangeutil.hpp"
#include "util/strprintf.hpp"

namespace arb {

template <typename Event>
class event_stream {
public:
    using size_type = arb_size_type;
    using index_type = arb_index_type;
    using event_type = Event;

    using event_time_type = ::arb::event_time_type<Event>;
    using event_data_type = ::arb::event_data_type<Event>;
    using event_kind_type = ::arb::event_kind_type<Event>;

    using state = event_stream_state<event_data_type>;

    event_stream() {}

    bool empty() const { return remaining_==0u; }

    size_type kind_count() const { return ev_kind_.size(); }

    size_type remaining_count() const { return remaining_; }

    size_type marked_count() const { return marked_; }

    void clear() {
        ev_data_.clear();
        ev_time_.clear();
        ev_kind_.clear();
        span_begin_vec_.clear();
        span_end_vec_.clear();
        offsets_.clear();
        remaining_ = 0;
        marked_ = 0;
    }

    // Initialize event stream from a vector of events, sorted by time.
    void init(const std::vector<Event>& staged) {
        using ::arb::event_time;
        using ::arb::event_data;
        using ::arb::event_kind;

        if (staged.size()>std::numeric_limits<size_type>::max()) {
            throw arbor_internal_error("event_stream: too many events for size type");
        }

        clear();

        util::assign_by(ev_data_, staged, [](const Event& ev) { return event_data(ev); });

        size_type i=0;
        for (const Event& ev : staged) {
            ev_time_.push_back(event_time(ev));
            if (auto k = event_kind(ev); ev_kind_.size()==0u || ev_kind_.back() < k) {
                ev_kind_.push_back(k);
                offsets_.push_back(i);
            }
            ++i;
        }
        span_begin_vec_.assign(offsets_.begin(), offsets_.end());
        span_end_vec_.assign(offsets_.begin(), offsets_.end());
        offsets_.push_back(i);

        remaining_ = i;
        arb_assert(remaining_ == staged.size());

        for (size_type k=0; k<ev_kind_.size(); ++k) {
            arb_assert(std::is_sorted(ev_time_.begin()+offsets_[k], ev_time_.begin()+offsets_[k+1]));
        }
    }

    // Designate for processing events `ev` at head of the event stream
    // until `event_time(ev)` > `t_until`.
    void mark_until_after(arb_value_type t_until) {
        for (size_type i=0; i<ev_kind_.size(); ++i) {
            const size_type end = offsets_[i+1];
            while (span_end_vec_[i]!=end && ev_time_[span_end_vec_[i]]<=t_until) {
                ++span_end_vec_[i];
            }
            marked_ += span_end_vec_[i] - span_begin_vec_[i];
        }
    }

    // Remove marked events from front of the event stream.
    void drop_marked_events() {
        for (size_type i=0; i<ev_kind_.size(); ++i) {
            span_begin_vec_[i] = span_end_vec_[i];
        }
        remaining_ -= marked_;
        marked_ = 0;
    }

    // Interface for access to marked events by mechanisms/kernels:
    state marked_events() const {
        return {ev_data_.data(), span_begin_vec_.data(), span_end_vec_.data(),
            (arb_size_type)ev_kind_.size(), marked_};
    }

    friend std::ostream& operator<<(std::ostream& out, const event_stream<Event>& m) {
        out << "[";
        for (arb_size_type k=0; k<m.ev_kind_.size(); ++k) {
            out << " { " << m.ev_kind_[k] << " : ";
            const auto a = m.offsets_[k];
            const auto b = m.span_begin_vec_[k];
            const auto c = m.span_end_vec_[k];
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
    std::vector<event_data_type> ev_data_;
    std::vector<event_kind_type> ev_kind_;
    std::vector<size_type> span_begin_vec_;
    std::vector<size_type> span_end_vec_;
    std::vector<size_type> offsets_;
    size_type remaining_ = 0;
    size_type marked_ = 0;
};

} // namespace arb

