#pragma once

// Indexed collection of pop-only event queues --- multicore back-end implementation.

#include <iosfwd>
#include <limits>
#include <utility>

#include <arbor/arbexcept.hpp>
#include <arbor/fvm_types.hpp>
#include <arbor/generic_event.hpp>

#include "backends/event.hpp"
#include "backends/event_stream_state.hpp"
#include "timestep_range.hpp"
#include "util/range.hpp"
#include "util/rangeutil.hpp"
#include "threading/threading.hpp"

namespace arb {
namespace multicore {

template <typename Event>
class event_stream {
public:
    using size_type = arb_size_type;
    using event_type = Event;
    using event_time_type = ::arb::event_time_type<Event>;
    using event_data_type = ::arb::event_data_type<Event>;
    using range = util::range<event_data_type*>;

    event_stream() = default;

    // returns true if the currently marked time step has no events
    bool empty() const {
        return ev_ranges_.empty() ||
               ev_data_.empty() ||
               !index_ ||
               index_ > ev_ranges_.size() ||
               !ev_ranges_[index_-1].size();
    }

    void clear() {
        ev_data_.clear();
        ev_ranges_.clear();
        index_ = 0;
    }

    // Initialize event streams from a vector of events, sorted by time.
    void init(const std::vector<Event>& staged, const timestep_range& dts) {
        using ::arb::event_time;

        if (staged.size()>std::numeric_limits<size_type>::max()) {
            throw arbor_internal_error("multicore/event_stream: too many events for size type");
        }

        // reset the state
        clear();

        // return if there are no time steps
        if (dts.empty()) return;

        // reserve space for events
        ev_data_.reserve(staged.size());
        ev_ranges_.reserve(dts.size());

        auto dt_first = dts.begin();
        const auto dt_last = dts.end();
        auto ev_first = staged.begin();
        const auto ev_last = staged.end();
        while(dt_first != dt_last) {
            // dereference iterator to current time step
            const auto dt = *dt_first;
            // add empty range for current time step
            auto ptr = ev_data_.data() + ev_data_.size();
            ev_ranges_.emplace_back(ptr, ptr);
            // loop over events
            for (; ev_first!=ev_last; ++ev_first) {
                const auto& ev = *ev_first;
                // check whether event falls within current timestep interval
                if (event_time(ev) >= dt.t_end()) break;
                // add event data and increase event range
                ev_data_.push_back(event_data(ev));
                ++ev_ranges_.back().right;
            }
            ++dt_first;
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
            return make_event_stream_state(ev_ranges_[index_-1].left, ev_ranges_[index_-1].right);
        }
    }

private:
    std::vector<event_data_type> ev_data_;
    std::vector<range> ev_ranges_;
    size_type index_ = 0;
};

} // namespace multicore
} // namespace arb
