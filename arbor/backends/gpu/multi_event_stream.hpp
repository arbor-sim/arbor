#pragma once

// Indexed collection of pop-only event queues --- CUDA back-end implementation.

#include <arbor/export.hpp>
#include <arbor/arbexcept.hpp>
#include <arbor/common_types.hpp>
#include <arbor/fvm_types.hpp>
#include <arbor/generic_event.hpp>

#include "backends/event.hpp"
#include "backends/multi_event_stream_state.hpp"
#include "memory/array.hpp"
#include "memory/copy.hpp"
#include "profile/profiler_macro.hpp"
#include "util/rangeutil.hpp"

namespace arb {
namespace gpu {

// Base class provides common implementations across event types.
class ARB_ARBOR_API multi_event_stream_base {
public:
    using size_type = cell_size_type;
    using value_type = arb_value_type;
    using index_type = arb_index_type;

    using array = memory::device_vector<value_type>;
    using iarray = memory::device_vector<index_type>;

    using const_view = array::const_view_type;
    using view = array::view_type;

    size_type n_streams() const { return n_stream_; }

    bool empty() const { return n_nonempty_stream_[0]==0; }

    void clear();

    // Designate for processing events `ev` at head of each event stream `i`
    // until `event_time(ev)` > `t_until[i]`.
    void mark_until_after(const_view t_until);

    // Designate for processing events `ev` at head of each event stream `i`
    // while `t_until[i]` > `event_time(ev)`.
    void mark_until(const_view t_until);

    // Remove marked events from front of each event stream.
    void drop_marked_events();

    // If the head of `i`th event stream exists and has time less than
    // `t_until[i]`, set `t_until[i]` to the event time.
    void event_time_if_before(view t_until);

protected:
    multi_event_stream_base() {}

    explicit multi_event_stream_base(size_type n_stream):
        n_stream_(n_stream),
        span_begin_(n_stream),
        span_end_(n_stream),
        mark_(n_stream),
        n_nonempty_stream_(1)
    {}

    // The list of events must be sorted sorted first by index and then by time.
    template <typename Event>
    void init(const std::vector<Event>& staged) {
        using ::arb::event_time;
        using ::arb::event_index;

        if (staged.size()>std::numeric_limits<size_type>::max()) {
            throw arbor_internal_error("gpu/multi_event_stream: too many events for size type");
        }

        arb_assert(util::is_sorted_by(staged, [](const Event& ev) { return event_index(ev); }));

        std::size_t n_ev = staged.size();
        tmp_ev_time_.clear();
        tmp_ev_time_.reserve(n_ev);

        util::assign_by(tmp_ev_time_, staged, [](const Event& ev) { return event_time(ev); });
        ev_time_ = array(memory::make_view(tmp_ev_time_));

        // Determine divisions by `event_index` in ev list.
        tmp_divs_.clear();
        tmp_divs_.reserve(n_stream_+1);

        size_type n_nonempty = 0;
        index_type ev_begin_i = 0;
        index_type ev_i = 0;
        tmp_divs_.push_back(ev_i);
        for (size_type s = 0; s<n_stream_; ++s) {
            while ((size_type)ev_i<n_ev && (size_type)event_index(staged[ev_i])<s+1) ++ev_i;

            // Within a subrange of events with the same index, events should
            // be sorted by time.
            arb_assert(util::is_sorted(util::subrange_view(tmp_ev_time_, ev_begin_i, ev_i)));
            n_nonempty += (tmp_divs_.back()!=ev_i);
            tmp_divs_.push_back(ev_i);
            ev_begin_i = ev_i;
        }

        arb_assert(tmp_divs_.size()==n_stream_+1);
        memory::copy(memory::make_view(tmp_divs_)(0,n_stream_), span_begin_);
        memory::copy(memory::make_view(tmp_divs_)(1,n_stream_+1), span_end_);
        memory::copy(span_begin_, mark_);
        n_nonempty_stream_[0] = n_nonempty;
    }

    size_type n_stream_;
    array ev_time_;
    iarray span_begin_;
    iarray span_end_;
    iarray mark_;
    iarray n_nonempty_stream_;

    // Host-side vectors for staging values in init():
    std::vector<value_type> tmp_ev_time_;
    std::vector<index_type> tmp_divs_;
};

template <typename Event>
class multi_event_stream: public multi_event_stream_base {
public:
    using event_data_type = ::arb::event_data_type<Event>;
    using data_array = memory::device_vector<event_data_type>;

    using state = multi_event_stream_state<event_data_type>;

    multi_event_stream() {}

    explicit multi_event_stream(size_type n_stream):
        multi_event_stream_base(n_stream) {}

    // Initialize event streams from a vector of events, sorted first by index
    // and then by time.
    void init(const std::vector<Event>& staged) {
        multi_event_stream_base::init(staged);

        tmp_ev_data_.clear();
        tmp_ev_data_.reserve(staged.size());

        using ::arb::event_data;
        util::assign_by(tmp_ev_data_, staged, [](const Event& ev) { return event_data(ev); });
        ev_data_ = data_array(memory::make_view(tmp_ev_data_));
    }

    state marked_events() const {
        return {n_stream_, ev_data_.data(), span_begin_.data(), mark_.data()};
    }

private:
    data_array ev_data_;

    // Host-side vector for staging event data in init():
    std::vector<event_data_type> tmp_ev_data_;
};

} // namespace gpu
} // namespace arb
