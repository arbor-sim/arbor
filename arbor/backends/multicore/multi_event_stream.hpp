#pragma once

#include <vector>

#include <arbor/arbexcept.hpp>
#include <arbor/fvm_types.hpp>
#include <arbor/generic_event.hpp>
#include <arbor/mechanism_abi.h>

#include "timestep_range.hpp"
#include "backends/event.hpp"
#include "util/range.hpp"
#include "util/rangeutil.hpp"
#include "util/span.hpp"

namespace arb {
namespace multicore {

class multi_event_stream {
public:
    using size_type = arb_size_type;
    using event_type = ::arb::deliverable_event;

    using event_time_type = ::arb::event_time_type<event_type>;
    using event_data_type = ::arb::event_data_type<event_type>;
    using event_index_type = ::arb::event_index_type<event_type>;

    multi_event_stream() = default;

    bool empty() const {
        return (index_ == 0 || index_ > num_dt_) || !ranges_[index_-1].size();
    }

    void clear() {
        num_dt_ = 0u;
        for (auto& v : ranges_) v.clear();
        ev_data_.clear();
        index_ = 0u;
    }

    void init(const std::vector<event_type>& staged, unsigned mech_id, arb_size_type n, const timestep_range& dts) {
        using ::arb::event_time;

        // reset the state
        clear();

        // return if there are no time steps
        if (dts.empty()) return;

        // resize ranges array
        num_dt_ = dts.size();
        if (ranges_.size() < num_dt_) ranges_.resize(num_dt_);

        // return if there are no events
        if (!n) return;

        // reserve space for events
        ev_data_.reserve(n);

        // loop over all events
        for (std::size_t i=0; i<staged.size(); ++i) {
            const auto& first = staged[i];
            // bail out if event is for antother mechanism
            if (first.handle.mech_id != mech_id) continue;
            const auto mech_index = first.handle.mech_index;
            // find all adjacent events with same index
            std::size_t s = i+1;
            while (s < staged.size() &&
                   staged[s].handle.mech_index == mech_index &&
                   staged[s].handle.mech_id == mech_id) {
                ++s;
            }
            // loop over timestep intervals
            for (size_type t=0; t<dts.size(); ++t) {
                const auto& dt = dts[t];
                // create empty event range
                arb_deliverable_event_range r{
                    mech_index,
                    size_type(ev_data_.size()),
                    size_type(ev_data_.size())};
                // loop over events with same index
                for (; i < s; ++i) {
                    const auto& ev = staged[i];
                    // check whether event falls within current timestep interval
                    if (event_time(ev) < dt.t1()) {
                        // add event data and increase event range
                        ev_data_.push_back(event_data(ev));
                        ++r.end;
                    }
                    else {
                        // bail out if event does not fall within current timestep interval
                        break;
                    }
                }
                // add event range if it is not empty
                if (r.end > r.begin) {
                    ranges_[t].push_back(r);
                }
                // bail out if all events have been used
                if (i >= s) break;
            }
            // reset loop variable
            i = s-1;
        }
        arb_assert(n == ev_data_.size());
    }

    void mark() {
        index_ += (index_ <= num_dt_ ? 1 : 0);
    }

    arb_deliverable_event_stream marked_events() const {
        if (empty()) return {0, nullptr, nullptr};
        return {
            size_type(ranges_[index_-1].size()),
            ev_data_.data(),
            ranges_[index_-1].data()
        };
    }

protected:
    size_type num_dt_ = 0u;
    std::vector<std::vector<arb_deliverable_event_range>> ranges_;
    std::vector<event_data_type> ev_data_;
    size_type index_ = 0u;
};

} // namespace multicore
} // namespace arb
