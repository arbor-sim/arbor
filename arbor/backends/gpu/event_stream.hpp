#pragma once

// Indexed collection of pop-only event queues --- CUDA back-end implementation.

#include "backends/event_stream_base.hpp"
#include "memory/memory.hpp"
#include "util/partition.hpp"
#include "util/range.hpp"
#include "util/rangeutil.hpp"
#include "util/transform.hpp"
#include "threading/threading.hpp"

namespace arb {
namespace gpu {

template <typename Event>
class event_stream : public event_stream_base<Event, typename memory::device_vector<::arb::event_data_type<Event>>::view_type> {
public:
    using base = event_stream_base<Event, typename memory::device_vector<::arb::event_data_type<Event>>::view_type>;
    using size_type = typename base::size_type;
    using event_data_type = typename base::event_data_type;
    using device_array = memory::device_vector<event_data_type>;

private: // members
    task_system_handle thread_pool_;
    device_array device_ev_data_;
    std::vector<size_type> offsets_;

public:
    event_stream() = default;
    event_stream(task_system_handle t): base(), thread_pool_{t} {}

    void clear() {
        base::clear();
        offsets_.clear();
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
        base::ev_spans_.resize(staged.size());
        base::ev_data_.resize(num_events);
        offsets_.resize(staged.size()+1);
        resize(device_ev_data_, num_events);

        // compute offsets by exclusive scan over staged events
        util::make_partition(offsets_,
            util::transform_view(staged, [&](const auto& v) { return v.size(); }),
            (size_type)0u);

        // assign, copy to device (and potentially sort) the event data in parallel
        arb_assert(thread_pool_);
        threading::parallel_for::apply(0, staged.size(), thread_pool_.get(),
            [this,&staged](size_type i) {
                const auto offset = offsets_[i];
                const auto size = staged[i].size();
                // add device range
                base::ev_spans_[i] = device_ev_data_(offset, offset + size);
                // host span
                auto host_span = memory::make_view(base::ev_data_)(offset, offset + size);
                // make event data and copy
                std::copy_n(util::transform_view(staged[i], [](const auto& x) {
                    return event_data(x);}).begin(), size, host_span.begin());
                // sort if necessary
                if constexpr (has_event_index<Event>::value) {
                    util::stable_sort_by(host_span, [](const event_data_type& ed) {
                        return event_index(ed); });
                }
                // copy to device
                memory::copy_async(host_span, base::ev_spans_[i]);
            });

        arb_assert(num_events == base::ev_data_.size());
    }

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
