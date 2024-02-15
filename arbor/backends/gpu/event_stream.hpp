#pragma once

// Indexed collection of pop-only event queues --- CUDA back-end implementation.

#include "arbor/spike_event.hpp"
#include "backends/event_stream_base.hpp"
#include "util/rangeutil.hpp"
#include "util/transform.hpp"
#include "util/partition.hpp"
#include "threading/threading.hpp"
#include "timestep_range.hpp"

#include <arbor/mechanism_abi.h>

ARB_SERDES_ENABLE_EXT(arb_deliverable_event_data, mech_index, weight);

namespace arb {
namespace gpu {

using event_lane_subrange = util::subrange_view_type<std::vector<pse_vector>>;

template <typename Event>
class event_stream: public event_stream_base<Event> {
public:
    using base = event_stream_base<Event>;
    using size_type = typename base::size_type;
    using event_data_type = typename base::event_data_type;
    using device_array = memory::device_vector<event_data_type>;

private: // members
    task_system_handle thread_pool_;
    device_array device_ev_data_;

public:
    event_stream() = default;
    event_stream(task_system_handle t): base(), thread_pool_{t} {}

    void clear() {
        base::clear();
    }

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
        base::ev_spans_.resize(staged.size() + 1);
        base::ev_data_.resize(num_events);
        resize(device_ev_data_, num_events);

        // compute offsets by exclusive scan over staged events
        util::make_partition(base::ev_spans_,
                             util::transform_view(staged, [&](const auto& v) { return v.size(); }),
                             (std::size_t)0u);

        // assign, copy to device (and potentially sort) the event data in parallel
        arb_assert(thread_pool_);
        threading::parallel_for::apply(0, staged.size(), thread_pool_.get(),
            [this, &staged](size_type i) {
                const auto beg = base::ev_spans_[i];
                const auto end = base::ev_spans_[i + 1];
                // host span
                auto host_span = memory::make_view(base::ev_data_)(beg, end);

                // make event data and copy
                std::copy_n(util::transform_view(staged[i],
                                                 [](const auto& x) { return event_data(x); }).begin(),
                            size,
                            host_span.begin());
                // sort if necessary
                if constexpr (has_event_index<Event>::value) {
                    util::stable_sort_by(host_span,
                                         [](const event_data_type& ed) { return event_index(ed); });
                }
                // copy to device
                memory::copy_async(host_span, base::ev_spans_[i]);
            });

        arb_assert(num_events == base::ev_data_.size());
    }

    static void multi_event_stream(const event_lane_subrange& lanes,
                                   const std::vector<target_handle>& handles,
                                   const std::vector<std::size_t>& divs,
                                   const timestep_range& steps,
                                   std::unordered_map<unsigned, event_stream>& streams) {

    }

    ARB_SERDES_ENABLE(event_stream<Event>, ev_data_, ev_spans_, device_ev_data_, index_);

private:
    template<typename D>
    static void resize(D& d, std::size_t size) {
        // resize if necessary
        if (d.size() < size) {
            d = D(size);
        }
    }
};

} // namespace gpu
} // namespace arb
