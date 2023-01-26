#pragma once

#include <vector>

#include <arbor/arbexcept.hpp>
#include <arbor/fvm_types.hpp>
#include <arbor/generic_event.hpp>
#include <arbor/mechanism_abi.h>

#include "event_map.hpp"
#include "timestep_range.hpp"
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

    //void init(const std::vector<event_type>& staged, const timestep_range& dts) {
    void init(const mechanism_event_map& staged, const timestep_range& dts) {
        using ::arb::event_time;

        clear();

        if (dts.empty()) return;
        num_dt_ = dts.size();

        const auto n = staged.aggregate_size();
        if (!n) return;
        ev_data_.reserve(n);

        if (ranges_.size() < num_dt_) ranges_.resize(num_dt_);
        // loop over all mech_index
        for (auto& [mech_index, vec] : staged) {
            // continue if no events found
            const auto s = vec.size();
            if (!s) continue;
            // loop over timestep intervals
            size_type i = 0;
            for (size_type t=0; t<dts.size(); ++t) {
                const auto& dt = dts[t];
                // create empty event range
                arb_deliverable_event_range r{
                    mech_index,
                    size_type(ev_data_.size()),
                    size_type(ev_data_.size())};
                // loop over remaining events
                for (; i<s; ++i) {
                    const auto& ev = vec[i];
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
                if (i>=s) break;
            }
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
