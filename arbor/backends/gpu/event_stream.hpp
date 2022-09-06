#pragma once

// Indexed collection of pop-only event queues --- CUDA back-end implementation.

#include <arbor/export.hpp>
#include <arbor/arbexcept.hpp>
#include <arbor/common_types.hpp>
#include <arbor/fvm_types.hpp>
#include <arbor/generic_event.hpp>

#include "backends/event.hpp"
#include "backends/event_stream_state.hpp"
#include "memory/array.hpp"
#include "memory/copy.hpp"
#include "profile/profiler_macro.hpp"
#include "util/rangeutil.hpp"
#include "util/strprintf.hpp"

namespace arb {
namespace gpu {

    /*
// Base class provides common implementations across event types.
class ARB_ARBOR_API event_stream_base {
public:
    using size_type = cell_size_type;
    using value_type = fvm_value_type;
    using index_type = fvm_index_type;

    using array = memory::device_vector<value_type>;
    using iarray = memory::device_vector<index_type>;

    using const_view = array::const_view_type;
    using view = array::view_type;

    bool empty() const { return span_begin_==(index_type)ev_data_.size(); }

    void clear();

    // Designate for processing events `ev` at head of the event stream
    // until `event_time(ev)` > `t_until`.
    void mark_until_after(const fvm_value_type& t_until);

    // Designate for processing events `ev` at head the stream
    // while `t_until` > `event_time(ev)`.
    void mark_until(const fvm_value_type& t_until);

    // Remove marked events from front of the event stream.
    void drop_marked_events() {
        span_begin_ = span_end_;
    }

protected:
    event_stream_base() {}

    // The list of events must be sorted sorted by time.
    template <typename Event>
    void init(const std::vector<Event>& staged) {
        using ::arb::event_time;

        if (staged.size()>std::numeric_limits<size_type>::max()) {
            throw arbor_internal_error("gpu/event_stream: too many events for size type");
        }

        std::size_t n_ev = staged.size();
        host_ev_time_.clear();
        host_ev_time_.reserve(n_ev);

        util::assign_by(host_ev_time_, staged, [](const Event& ev) { return event_time(ev); });
        ev_time_ = array(memory::make_view(host_ev_time_));
        span_begin_ = span_end_ = 0;
    }

    array ev_time_;
    index_type span_begin_ = 0;
    index_type span_end_ = 0;

    // Host-side vectors for staging values in init():
    std::vector<value_type> host_ev_time_;
};

template <typename Event>
class event_stream: public event_stream_base {
public:
    using event_data_type = ::arb::event_data_type<Event>;
    using data_array = memory::device_vector<event_data_type>;

    using state = event_stream_state<event_data_type>;

    event_stream() {}

    // Initialize event streams from a vector of events, sorted by time.
    void init(const std::vector<Event>& staged) {
        event_stream_base::init(staged);

        tmp_ev_data_.clear();
        tmp_ev_data_.reserve(staged.size());

        using ::arb::event_data;
        util::assign_by(tmp_ev_data_, staged, [](const Event& ev) { return event_data(ev); });
        ev_data_ = data_array(memory::make_view(tmp_ev_data_));
    }

    state marked_events() const {
        //std::cout << "RETURNING: " << span_begin_ << "--" << span_end_ << "\n";
        return {ev_data_.data()+span_begin_, ev_data_.data()+span_end_};
    }

private:
    data_array ev_data_;

    // Host-side vector for staging event data in init():
    std::vector<event_data_type> tmp_ev_data_;
};
    */

template <typename Event>
class event_stream {
public:
    using size_type = arb_size_type;
    using index_type = arb_index_type;
    using event_type = Event;

    using event_time_type = ::arb::event_time_type<Event>;
    using event_data_type = ::arb::event_data_type<Event>;

    using state = event_stream_state<event_data_type>;

    event_stream() {}

    bool empty() const { return span_begin_==(index_type)num_events_; }

    void clear() {
        num_events_ = 0;
        span_begin_ = 0;
        span_end_ = 0;
    }

    // Initialize event streams from a vector of events, sorted by time.
    void init(std::vector<Event> staged) {
        using ::arb::event_time;
        using ::arb::event_data;

        if (staged.size()>std::numeric_limits<size_type>::max()) {
            throw arbor_internal_error("multicore/event_stream: too many events for size type");
        }

        num_events_ = staged.size();

        util::assign_by(host_ev_time_, staged,
                [](const Event& ev) {return event_time(ev); });
        util::assign_by(host_ev_data_, staged,
                [](const Event& ev) {return event_data(ev); });
        ev_time_ = memory::make_view(host_ev_time_);
        ev_data_ = memory::make_view(host_ev_data_);
        span_begin_ = span_end_ = 0;
    }

    // Designate for processing events `ev` at head of the event stream
    // until `event_time(ev)` > `t_until`.
    void mark_until_after(const arb_value_type& t_until) {
        using ::arb::event_time;

        const index_type end = ev_time_.size();
        while (span_end_!=end && !(host_ev_time_[span_end_]>t_until)) {
            ++span_end_;
        }
    }

    // Designate for processing events `ev` at head the stream
    // while `t_until` > `event_time(ev)`.
    void mark_until(const arb::arb_value_type& t_until) {
        using ::arb::event_time;

        const index_type end = ev_time_.size();
        while (span_end_!=end && t_until>host_ev_time_[span_end_]) {
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

    friend std::ostream& operator<<(std::ostream& out, const event_stream<Event>& m) {
        int n_ev = m.ev_data_.size();

        out << "[";

        for (int ev_i = 0; ev_i<n_ev; ++ev_i) {
            bool discarded = ev_i<m.span_begin_;
            bool marked = !discarded && ev_i<m.span_end_;

            if (discarded) {
                out << "        x";
            }
            else {
                out << util::strprintf(" % 7.3f%c", m.host_ev_time_[ev_i]-1, marked?'*':' ');
            }
        }
        return out << "]";
    }

private:
    memory::device_vector<arb_value_type> ev_time_;
    memory::device_vector<event_data_type> ev_data_;

    // Host-side vectors for staging values in init():
    std::vector<arb_value_type> host_ev_time_;
    std::vector<event_data_type> host_ev_data_;

    index_type span_begin_ = 0;
    index_type span_end_ = 0;
    index_type num_events_ = 0;
};

} // namespace gpu
} // namespace arb
