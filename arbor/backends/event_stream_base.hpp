#pragma once

#include <vector>

#include <arbor/mechanism_abi.h>

#include "backends/event.hpp"
#include "backends/event_stream_state.hpp"
#include "event_lane.hpp"
#include "timestep_range.hpp"

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

struct spike_event_stream_base: event_stream_base<deliverable_event> {
    // Take in one event lane per cell `gid` and reorganise into one stream per
    // synapse `mech_id`.
    //
    // - Due to the cell group coalescing multiple cells and their synapses into
    //   one object, one `mech_id` can touch multiple lanes / `gid`s.
    // - Inversely, two `mech_id`s can cover different, but overlapping sets of `gid`s
    // - Multiple `mech_id`s can receive events from the same source
    //
    // Pre:
    // - Events in `lanes[ix]` forall ix
    //   * are sorted by time
    //   * `ix` maps to exactly one cell in the local cell group
    // - `divs` partitions `handles` such that the target handles for cell `ix`
    //   are located in `handles[divs[ix]..divs[ix + 1]]`
    // - `handles` records `(mech_id, index)` of a target s.t. `index` is the instance
    //   with the set identified by `mech_id`, e.g. a single synapse placed on a multi-
    //   location locset (plus the merging across cells by groups)
    // Post:
    // - streams[mech_id] contains a list of all events for synapse `mech_id` s.t.
    //   * the list is sorted by (time_step, lid, time)
    //   * the list is partitioned by `time_step` via `ev_spans`
    template<typename EventStream>
    friend void initialize(const event_lane_subrange& lanes,
                           const std::vector<target_handle>& handles,
                           const std::vector<std::size_t>& divs,
                           const timestep_range& steps,
                           std::unordered_map<unsigned, EventStream>& streams) {
        arb_assert(lanes.size() < divs.size());

        // reset streams and allocate sufficient space for temporaries
        auto n_steps = steps.size();
        for (auto& [id, stream]: streams) {
            stream.clear();
            stream.ev_spans_.resize(steps.size() + 1, 0);
            stream.spikes_.clear();
            // ev_data_ has been cleared during v.clear(), so we use its capacity
            stream.spikes_.reserve(stream.ev_data_.capacity());
        }

        // loop over lanes: group events by mechanism and sort them by time
        arb_size_type cell = 0;
        for (const auto& lane: lanes) {
            auto div = divs[cell];
            arb_size_type step = 0;
            for (const auto& evt: lane) {
                step = std::lower_bound(steps.begin() + step,
                                        steps.end(),
                                        evt.time,
                                        [](const auto& bucket, time_type time) { return bucket.t_end() <= time; })
                     - steps.begin();
                // Events coinciding with epoch's upper boundary belong to next epoch
                if (step >= n_steps) break;
                arb_assert(div + evt.target < handles.size());
                const auto& handle = handles[div + evt.target];
                auto& stream = streams[handle.mech_id];
                stream.spikes_.emplace_back(spike_data{step, handle.mech_index, evt.time, evt.weight});
                stream.ev_spans_[step + 1]++;
            }
            ++cell;
        }

        // TODO parallelise over streams, however, need a proper testcase/benchmark.
        // auto tg = threading::task_group(ts.get());
        for (auto& [id, stream]: streams) {
            // tg.run([&stream=stream]() {
                // scan to partition stream (aliasing is explicitly allowed)
                std::inclusive_scan(stream.ev_spans_.begin(), stream.ev_spans_.end(),
                                    stream.ev_spans_.begin());
                // This is made slightly faster by using pdqsort.
                // Despite events being sorted by time in the partitions defined
                // by the lane index here, they are not _totally_ sorted, thus
                // sort is needed, merge not being strong enough :/
                std::sort(stream.spikes_.begin(), stream.spikes_.end());
                // copy temporary deliverable_events into stream's ev_data_
                stream.ev_data_.reserve(stream.spikes_.size());
                for (const auto& spike: stream.spikes_) stream.ev_data_.emplace_back(event_data_type{spike.mech_index, spike.weight});
                // delegate to derived class init: static cast necessary to access protected init()
                static_cast<spike_event_stream_base&>(stream).init();
            // });
        }
        // tg.wait();
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
