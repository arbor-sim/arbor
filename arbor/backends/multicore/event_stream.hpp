#pragma once

// Indexed collection of pop-only event queues --- multicore back-end implementation.

#include "arbor/spike_event.hpp"
#include "backends/event_stream_base.hpp"
#include "timestep_range.hpp"
#include "util/partition.hpp"
#include "util/rangeutil.hpp"

namespace arb {
namespace multicore {


using event_lane_subrange = util::subrange_view_type<std::vector<pse_vector>>;

template <typename Event>
struct event_stream: public event_stream_base<Event> {
    using base = event_stream_base<Event>;
    using size_type = typename base::size_type;

    event_stream() = default;

    using base::clear;

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
        base::ev_spans_.reserve(staged.size() + 1);
        base::ev_data_.reserve(num_events);

        // add event data and spans
        for (const auto& v : staged) {
            for (const auto& ev: v) base::ev_data_.push_back(event_data(ev));
            base::ev_spans_.push_back(base::ev_data_.size());
        }

        arb_assert(num_events == base::ev_data_.size());
        arb_assert(staged.size() + 1 == base::ev_spans_.size());
    }

    ARB_SERDES_ENABLE(event_stream<Event>, ev_data_, ev_spans_, index_);

    // Construct a mapping of mech_id to a stream s.t. streams are partitioned into
    // time step buckets by `ev_span`
    static auto
    multi_event_stream(const event_lane_subrange& lanes,
                       const std::vector<target_handle>& handles,
                       const std::vector<std::size_t>& divs,
                       const timestep_range& steps,
                       std::unordered_map<unsigned, event_stream>& streams) {
        auto n_streams = streams.size();
        auto n_steps = steps.size();

        std::unordered_map<unsigned, std::vector<std::size_t>> dt_sizes;
        for (auto& [k, v]: streams) dt_sizes[k] = std::vector<std::size_t>(n_steps, 0);

        auto cell = 0;
        for (auto& lane: lanes) {
            auto div = divs[cell];
            arb_size_type step = 0;
            for (auto evt: lane) {
                auto time = evt.time;
                auto weight = evt.weight;
                auto target = evt.target;
                while(step < n_steps && time >= steps[step].t_end()) ++step;
                // Events coinciding with epoch's upper boundary belong to next epoch
                if (step >= n_steps) break;
                if (div + target > handles.size()) throw std::out_of_range("target handle index out of range");
                auto& handle = handles[div + target];
                if (!streams.count(handle.mech_id)) throw std::out_of_range("streams: mech id unknown");
                streams[handle.mech_id].ev_data_.push_back({handle.mech_index, weight});
                if (!dt_sizes.count(handle.mech_id)) throw std::out_of_range("dt: mech id unknown");
                if (step >= dt_sizes[handle.mech_id].size()) throw std::out_of_range("step oob");
                dt_sizes[handle.mech_id][step]++;
            }
            ++cell;
        }

        for (auto& [k, v]: streams) util::make_partition(v.ev_spans_, dt_sizes[k]);

        return streams;
    }
};
} // namespace multicore
} // namespace arb
