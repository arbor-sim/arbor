#pragma once

#include <vector>

#include <arbor/arbexcept.hpp>
#include <arbor/fvm_types.hpp>
#include <arbor/generic_event.hpp>
#include <arbor/mechanism_abi.h>

#include "backends/event.hpp"
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
        return ev_data_.empty() || index_ >= num_events_.size();
    }

    void clear() {
        num_streams_ = 0u;
        num_dt_ = 0u;
        ranges_.clear();
        ranges_tmp_.clear();
        num_events_.clear();
        ev_data_.clear();
        index_ = 0u;
    }

    void init(const std::vector<event_type>& staged, const timestep_range& dts) {
        using ::arb::event_time;

        if (staged.size()>std::numeric_limits<size_type>::max()) {
            throw arbor_internal_error("multicore/event_stream: too many events for size type");
        }

        clear();

        if (dts.empty()) return;
        num_dt_ = dts.size();
        num_events_.assign(num_dt_+1, 0u);

        if (staged.empty()) return;
        ev_data_.reserve(staged.size());

        auto divide = [this, &staged, &dts] (event_index_type idx, size_type begin, size_type end) {
            for (auto dt : dts) {
                ranges_tmp_.push_back(arb_deliverable_event_range{idx,begin,begin});
                for (; begin<end; ++begin) {
                    const auto& ev = staged[begin];
                    if (event_time(ev) < dt.t1()) {
                        ++ranges_tmp_.back().end;
                    }
                    else {
                        break;
                    }
                }
            }
        };

        num_streams_ = 1u;
        size_type old_i = 0u;
        auto old_index = event_index(staged[0]);
        for (size_type i: util::count_along(staged)) {
            const auto& ev = staged[i];
            ev_data_.push_back(event_data(ev));
            const auto new_index = event_index(ev);
            if (new_index != old_index) {
                divide(old_index, old_i, i);
                ++num_streams_;
                old_index = new_index;
                old_i = i;
            }
        }
        divide(old_index, old_i, staged.size());

        arb_assert(num_streams_*num_dt_ == ranges_tmp_.size());
        arb_assert(staged.size() == ev_data_.size());

        // transpose and add empty row at begin
        ranges_.reserve(ranges_tmp_.size() + num_streams_);
        for (size_type s = 0u; s < num_streams_; ++s) {
            ranges_.push_back(arb_deliverable_event_range{0,0,0});
        }
        for (size_type t = 0u; t < num_dt_; ++t) {
            size_type n = 0u;
            for (size_type s = 0u; s < num_streams_; ++s) {
                const auto& r = ranges_tmp_[s*num_dt_+t];
                ranges_.push_back(r);
                n += r.end - r.begin;
            }
            num_events_[t+1] = n;
        }
    }
    
    void mark() {
        index_ += (index_ < num_dt_ ? 1 : 0);
    }

    arb_deliverable_event_stream marked_events() const {
        return {
            num_streams_,
            empty()? 0u : num_events_[index_],
            ev_data_.data(),
            ranges_.data() + index_*num_streams_
        };
    }

protected:
    size_type num_streams_ = 0u;
    size_type num_dt_ = 0u;
    std::vector<arb_deliverable_event_range> ranges_;
    std::vector<arb_deliverable_event_range> ranges_tmp_;
    std::vector<size_type> num_events_;
    std::vector<event_data_type> ev_data_;
    size_type index_ = 0u;
};

} // namespace multicore
} // namespace arb
