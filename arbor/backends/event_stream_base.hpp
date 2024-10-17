#pragma once

#include <vector>

#include <arbor/generic_event.hpp>
#include <arbor/mechanism_abi.h>


#include "backends/event.hpp"
#include "backends/event_stream_state.hpp"
#include "event_lane.hpp"
#include "timestep_range.hpp"
#include "util/partition.hpp"

ARB_SERDES_ENABLE_EXT(arb_deliverable_event_data, mech_index, weight);

namespace arb {

template <typename Event>
struct event_stream_base {
    using size_type = std::size_t;
    using event_type = Event;
    using event_time_type = ::arb::event_time_type<Event>;
    using event_data_type = ::arb::event_data_type<Event>;

protected: // members
    std::vector<event_data_type> ev_data_;
    std::vector<std::size_t> ev_spans_ = {0};
    size_type index_ = 0;
    event_data_type* base_ptr_ = nullptr;

public:
    event_stream_base() = default;

    // returns true if the currently marked time step has no events
    bool empty() const {
        return ev_data_.empty()                          // No events
            || index_ < 1                                // Since we index with a left bias, index_ must be at least 1
            || index_ >= ev_spans_.size()                // Cannot index at container length
            || ev_spans_[index_-1] >= ev_spans_[index_]; // Current span is empty
    }

    void mark() { index_ += 1; }

    auto marked_events() {
        auto beg = (event_data_type*)nullptr;
        auto end = (event_data_type*)nullptr;
        if (!empty()) {
            beg = base_ptr_ + ev_spans_[index_-1];
            end = base_ptr_ + ev_spans_[index_];
        }
        return make_event_stream_state(beg, end);
    }

    // clear all previous data
    void clear() {
        ev_data_.clear();
        // Clear + push doesn't allocate a new vector
        ev_spans_.clear();
        ev_spans_.push_back(0);
        base_ptr_ = nullptr;
        index_ = 0;
    }

    // Construct a mapping of mech_id to a stream s.t. streams are partitioned into
    // time step buckets by `ev_span`
    template<typename EventStream>
    static std::enable_if_t<std::is_base_of_v<event_stream_base<deliverable_event>, EventStream>>
    multi_event_stream(const event_lane_subrange& lanes,
                       const std::vector<target_handle>& handles,
                       const std::vector<std::size_t>& divs,
                       const timestep_range& steps,
                       std::unordered_map<unsigned, EventStream>& streams) {
        arb_assert(lanes.size() < divs.size());

        // temporary data structures to hold
        // - the number of events in every time interval per mechanism
        // - time sorted events per mechanism
        std::unordered_map<unsigned, std::vector<std::size_t>> dt_sizes;
        std::unordered_map<unsigned, std::vector<deliverable_event>> evts;

        // reset streams and allocate sufficient space for temporaries
        auto n_steps = steps.size();
        for (auto& [k, v]: streams) {
            v.clear();
            dt_sizes[k].resize(n_steps, 0);
            evts[k].reserve(v.ev_data_.capacity());
        }

        // loop over lanes: group events by mechanism and sort them by time
        auto cell = 0;
        for (const auto& lane: lanes) {
            auto div = divs[cell];
            ++cell;
            arb_size_type step = 0;
            for (const auto& evt: lane) {
                auto time = evt.time;
                auto weight = evt.weight;
                auto target = evt.target;
                while(step < n_steps && time >= steps[step].t_end()) ++step;
                // Events coinciding with epoch's upper boundary belong to next epoch
                if (step >= n_steps) break;
                arb_assert(div + target < handles.size());
                const auto& handle = handles[div + target];
                auto& sorted_evts = evts[handle.mech_id];
                sorted_evts.emplace_back(time, handle, weight);
                // insertion sort with last element as pivot
                auto first = sorted_evts.begin();
                auto last = sorted_evts.end();
                auto pivot = std::prev(last, 1);
                std::rotate(std::upper_bound(first, pivot, *pivot, [](auto const& l, auto const& r) noexcept { return l.time < r.time; }),
                    pivot, last);
                // increment count in current time interval
                dt_sizes[handle.mech_id][step]++;
            }
        }

        for (auto& [id, stream]: streams) {
            // copy temporary deliverable_events into stream's ev_data_
            auto& sorted_evts = evts[id];
            stream.ev_data_.reserve(sorted_evts.size());
            std::transform(sorted_evts.begin(), sorted_evts.end(), std::back_inserter(stream.ev_data_),
                [](auto const& e) noexcept -> arb_deliverable_event_data { return {e.handle.mech_index, e.weight}; });
            // scan over dt_sizes[id] written to ev_spans_
            util::make_partition(stream.ev_spans_, dt_sizes[id]);
            stream.init();
        }
    }
};

} // namespace arb
