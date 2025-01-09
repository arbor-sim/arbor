#pragma once

#include <vector>

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
    using event_type = Event;
    using event_data_type = decltype(event_data(std::declval<Event>()));

protected: // members
    std::vector<event_data_type> ev_data_;
    std::vector<std::size_t> ev_spans_ = {0};
    std::size_t index_ = 0;
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

  protected:
    // backend specific initializations
    virtual void init() = 0;
};

struct spike_event_stream_base : event_stream_base<deliverable_event> {
    template<typename EventStream>
    friend void initialize(const event_lane_subrange& lanes,
                           const std::vector<target_handle>& handles,
                           const std::vector<std::size_t>& divs,
                           const timestep_range& steps,
                           std::unordered_map<unsigned, EventStream>& streams) {
        arb_assert(lanes.size() < divs.size());

        // reset streams and allocate sufficient space for temporaries
        auto n_steps = steps.size();
        for (auto& [k, v]: streams) {
            v.clear();
            v.spike_counter_.clear();
            v.spike_counter_.resize(steps.size(), 0);
            v.spikes_.clear();
            // ev_data_ has been cleared during v.clear(), so we use its capacity
            v.spikes_.reserve(v.ev_data_.capacity());
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
                auto& stream = streams[handle.mech_id];
                stream.spikes_.push_back(spike_data{step, handle.mech_index, time, weight});
                // insertion sort with last element as pivot
                // ordering: first w.r.t. step, within a step: mech_index, within a mech_index: time
                auto first = stream.spikes_.begin();
                auto last = stream.spikes_.end();
                auto pivot = std::prev(last, 1);
                std::rotate(std::upper_bound(first, pivot, *pivot), pivot, last);
                // increment count in current time interval
                stream.spike_counter_[step]++;
            }
        }

        for (auto& [id, stream]: streams) {
            // copy temporary deliverable_events into stream's ev_data_
            stream.ev_data_.reserve(stream.spikes_.size());
            std::transform(stream.spikes_.begin(), stream.spikes_.end(), std::back_inserter(stream.ev_data_),
                [](auto const& e) noexcept -> arb_deliverable_event_data {
                return {e.mech_index, e.weight}; });
            // scan over spike_counter_ and  written to ev_spans_
            util::make_partition(stream.ev_spans_, stream.spike_counter_);
            // delegate to derived class init: static cast necessary to access protected init()
            static_cast<spike_event_stream_base&>(stream).init();
        }
    }

  protected: // members
    struct spike_data {
        arb_size_type step = 0;
        cell_local_size_type mech_index = 0;
        time_type time = 0;
        float weight = 0;
        auto operator<=>(spike_data const&) const noexcept = default;
    };
    std::vector<spike_data> spikes_;
    std::vector<std::size_t> spike_counter_;
};

struct sample_event_stream_base : event_stream_base<sample_event> {
    friend void initialize(const std::vector<std::vector<sample_event>>& staged,
                           sample_event_stream_base& stream) {
        // clear previous data
        stream.clear();

        // return if there are no timestep bins
        if (!staged.size()) return;

        // return if there are no events
        auto num_events = util::sum_by(staged, [] (const auto& v) {return v.size();});
        if (!num_events) return;

        // allocate space for spans and data
        stream.ev_spans_.reserve(staged.size() + 1);
        stream.ev_data_.reserve(num_events);

        // add event data and spans
        for (const auto& v : staged) {
            for (const auto& ev: v) stream.ev_data_.push_back(ev.raw);
            stream.ev_spans_.push_back(stream.ev_data_.size());
        }

        arb_assert(num_events == stream.ev_data_.size());
        arb_assert(staged.size() + 1 == stream.ev_spans_.size());
        stream.init();
    }
};

} // namespace arb
