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
#include "util/strprintf.hpp"

namespace arb {
namespace multicore {

template <typename Event>
class event_stream {
public:
    using size_type = arb_size_type;
    using event_type = Event;

    using event_time_type = ::arb::event_time_type<Event>;
    using event_data_type = ::arb::event_data_type<Event>;

    using state = event_stream_state<event_data_type>;

    event_stream() = default;

    bool empty() const { return ev_data_.empty() || index_ >= offsets_.size(); }

    void clear() {
        tmp_.clear();
        ev_data_.clear();
        offsets_.clear();
        index_ = 0;
    }

    // Initialize event streams from a vector of events, sorted by time.
    void init(const std::vector<Event>& staged, const timestep_range& dts) {
        using ::arb::event_time;

        if (staged.size()>std::numeric_limits<size_type>::max()) {
            throw arbor_internal_error("multicore/event_stream: too many events for size type");
        }

        auto append_tmp = [this]() {
            if (tmp_.empty()) return;
            if constexpr (has_event_index<Event>::value) {
                util::stable_sort_by(tmp_, [](const Event& ev) { return event_index(ev); });
            }
            for (const auto ev_ : tmp_) {
                ev_data_.push_back(event_data(ev_));
            }
            tmp_.clear();
        };

        clear();
        ev_data_.reserve(staged.size());
        offsets_.assign(dts.size()+1, staged.size());
        offsets_[0] = 0u;
        size_type dt_index = 0u;
        auto dt = dts[dt_index];
        for (const Event& ev : staged) {
            const auto ev_t = event_time(ev);
            if (ev_t >= dt.t_end()) {
                while (ev_t >= dt.t_end()) {
                    offsets_[++dt_index] = ev_data_.size()+tmp_.size();
                    dt = dts[dt_index];
                }
                append_tmp();
            }
            tmp_.push_back(ev);
        }
        append_tmp();

        arb_assert(ev_data_.size() == staged.size());
    }

    void mark() {
        index_ += (index_ < offsets_.size() ? 1 : 0);
    }

    state marked_events() const {
        if (empty()) return {nullptr, nullptr};
        return {ev_data_.data()+offsets_[index_-1], ev_data_.data()+offsets_[index_]};
    }

protected:
    std::vector<Event> tmp_;
    std::vector<event_data_type> ev_data_;
    std::vector<size_type> offsets_;
    size_type index_ = 0;
};

} // namespace multicore
} // namespace arb
