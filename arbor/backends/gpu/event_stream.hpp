#pragma once

// Indexed collection of pop-only event queues --- CUDA back-end implementation.

#include <iosfwd>
#include <limits>
#include <utility>

#include <arbor/arbexcept.hpp>
#include <arbor/fvm_types.hpp>
#include <arbor/generic_event.hpp>

#include "backends/event.hpp"
#include "backends/event_stream_state.hpp"
#include "memory/memory.hpp"
#include "timestep_range.hpp"
#include "util/range.hpp"
#include "util/rangeutil.hpp"
#include "threading/threading.hpp"

namespace arb {
namespace gpu {

template <typename Event>
class event_stream {
public:
    using size_type = arb_size_type;
    using event_type = Event;
    using event_time_type = ::arb::event_time_type<Event>;
    using event_data_type = ::arb::event_data_type<Event>;

    event_stream() = default;
    event_stream(task_system_handle t): thread_pool_{t} {}

    // returns true if the currently marked time step has no events
    bool empty() const {
        return ev_ranges_.empty() ||
               ev_data_.empty() ||
               !(index_) ||
               index_ > ev_ranges_.size() ||
               !(ev_ranges_[index_-1].second - ev_ranges_[index_-1].first);
    }

    void clear() {
        ev_data_.clear();
        ev_ranges_.clear();
        index_ = 0;
    }

    void init(const std::vector<Event>& staged, const timestep_range& dts) {
        using ::arb::event_time;

        if (staged.size()>std::numeric_limits<size_type>::max()) {
            throw arbor_internal_error("multicore/event_stream: too many events for size type");
        }

        // reset the state
        clear();

        // return if there are no time steps
        if (dts.empty()) return;

        // set up task group
        arb_assert(thread_pool_);
        threading::task_group g(thread_pool_.get());

        // reserve space for events
        ev_data_.reserve(staged.size());
        ev_ranges_.reserve(dts.size());

        // resize GPU data
        resize(device_ev_data_, staged.size());

        auto dt_first = dts.begin();
        const auto dt_last = dts.end();
        auto ev_first = staged.begin();
        const auto ev_last = staged.end();
        while(dt_first != dt_last) {
            // dereference iterator to current time step
            const auto dt = *dt_first;
            // add empty range for current time step
            ev_ranges_.emplace_back(ev_data_.size(), ev_data_.size());
            // loop over events
            for (; ev_first!=ev_last; ++ev_first) {
                const auto& ev = *ev_first;
                // check whether event falls within current timestep interval
                if (event_time(ev) >= dt.t_end()) break;
                // add event data and increase event range
                ev_data_.push_back(event_data(ev));
                ++ev_ranges_.back().second;
            }
            // if we use event indices: stable sort the range before copying to GPU
            if constexpr (has_event_index<Event>::value) {
                static_assert(std::is_same<Event, deliverable_event>::value);
                g.run([this]() {
                    const auto [first, last] = ev_ranges_.back();
                    util::stable_sort_by(util::make_range(ev_data_.data() + first, ev_data_.data() + last),
                            [](const event_data_type& ed) { return event_index(ed); });
                    copy(ev_data_, device_ev_data_, first, last);
                });

            }
            ++dt_first;
        }

        if constexpr (has_event_index<Event>::value) {
            // wait for sorting and copying to be done
            g.wait();
        }
        else {
            // if we don't use event indices: copy to GPU in one go
            static_assert(std::is_same<Event, sample_event>::value);
            copy(ev_data_, device_ev_data_);
        }
        arb_assert(ev_data_.size() == staged.size());
    }

    void mark() {
        index_ += (index_ <= ev_ranges_.size() ? 1 : 0);
    }

    auto marked_events() {
        if (empty()) {
            return make_event_stream_state((event_data_type*)nullptr, (event_data_type*)nullptr);
        } else {
            auto ptr = device_ev_data_.data();
            const auto [first, last] = ev_ranges_[index_-1];
            return make_event_stream_state(ptr + first, ptr + last);
        }
    }

private:
    template<typename D>
    static void resize(D& d, std::size_t size) {
        // resize if necessary
        if (d.size() < size) {
            d = D(size);
        }
    }

    template<typename H, typename D>
    static void copy(const H& h, D& d) {
        memory::copy_async(h, memory::make_view(d)(0u, h.size()));
    }

    template<typename H, typename D>
    static void copy(const H& h, D& d, size_type first, size_type last) {
        memory::copy_async(memory::make_const_view(h)(first, last), memory::make_view(d)(first, last));
    }

private:
    task_system_handle thread_pool_;
    std::vector<event_data_type> ev_data_;
    memory::device_vector<event_data_type> device_ev_data_;
    std::vector<std::pair<size_type,size_type>> ev_ranges_;
    size_type index_ = 0;
};

} // namespace gpu
} // namespace arb
