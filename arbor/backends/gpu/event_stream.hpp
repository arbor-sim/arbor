#pragma once

// Indexed collection of pop-only event queues --- CUDA back-end implementation.

#include <arbor/mechanism_abi.h>

#include "backends/event_stream_base.hpp"
#include "util/transform.hpp"
#include "threading/threading.hpp"
#include "timestep_range.hpp"
#include "memory/memory.hpp"

namespace arb {
namespace gpu {

template <typename Event>
struct event_stream: public event_stream_base<Event> {
public:
    using base = event_stream_base<Event>;
    using size_type = typename base::size_type;
    using event_data_type = typename base::event_data_type;
    using device_array = memory::device_vector<event_data_type>;

    using base::clear;
    using base::ev_data_;
    using base::ev_spans_;
    using base::base_ptr_;

    event_stream() = default;
    event_stream(task_system_handle t): base(), thread_pool_{t} {}

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
        ev_spans_.resize(staged.size() + 1);
        ev_data_.resize(num_events);
        resize(device_ev_data_, num_events);

        // compute offsets by exclusive scan over staged events
        util::make_partition(ev_spans_,
                             util::transform_view(staged, [](const auto& v) { return v.size(); }),
                             0ull);

        // assign, copy to device (and potentially sort) the event data in parallel
        arb_assert(thread_pool_);
        arb_assert(ev_spans_.size() == staged.size() + 1);
        threading::parallel_for::apply(0, ev_spans_.size() - 1, thread_pool_.get(),
            [this, &staged](size_type i) {
                const auto beg = ev_spans_[i];
                const auto end = ev_spans_[i + 1];
                arb_assert(end >= beg);
                const auto len = end - beg;

                auto host_span = memory::make_view(ev_data_)(beg, end);

                // make event data and copy
                std::copy_n(util::transform_view(staged[i],
                                                 [](const auto& x) { return event_data(x); }).begin(),
                            len,
                            host_span.begin());
                // sort if necessary
                if constexpr (has_event_index<Event>::value) {
                    util::stable_sort_by(host_span,
                                         [](const event_data_type& ed) { return event_index(ed); });
                }
                // copy to device
                auto device_span = memory::make_view(device_ev_data_)(beg, end);
                memory::copy_async(host_span, device_span);
            });

        base_ptr_ = device_ev_data_.data();

        arb_assert(num_events == device_ev_data_.size());
        arb_assert(num_events == ev_data_.size());
    }

    // Initialize event stream assuming ev_data_ and ev_span_ has
    // been set previously (e.g. by `base::multi_event_stream`)
    void init() {
        resize(device_ev_data_, ev_data_.size());
        base_ptr_ = device_ev_data_.data();

        threading::parallel_for::apply(0, ev_spans_.size() - 1, thread_pool_.get(),
           [this](size_type i) {
               const auto beg = ev_spans_[i];
               const auto end = ev_spans_[i + 1];
               arb_assert(end >= beg);

               auto host_span = memory::make_view(ev_data_)(beg, end);
               auto device_span = memory::make_view(device_ev_data_)(beg, end);

               // sort if necessary
               if constexpr (has_event_index<Event>::value) {
                   util::stable_sort_by(host_span,
                                        [](const event_data_type& ed) { return event_index(ed); });
               }
               // copy to device
               memory::copy_async(host_span, device_span);
           });
    }

    template<typename D>
    static void resize(D& d, std::size_t size) {
        // resize if necessary
        if (d.size() < size) {
            d = D(size);
        }
    }

    ARB_SERDES_ENABLE(event_stream<Event>,
                      ev_data_,
                      ev_spans_,
                      device_ev_data_,
                      index_);

    task_system_handle thread_pool_;
    device_array device_ev_data_;
};

} // namespace gpu
} // namespace arb
