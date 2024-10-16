#pragma once

#include <vector>
#include <random>
#include <gtest/gtest.h>

#include "timestep_range.hpp"
#include "backends/event.hpp"
#include "util/rangeutil.hpp"

namespace {

using namespace arb;

void check_result(arb_deliverable_event_data const* results, std::vector<arb_deliverable_event_data> const& expected) {
    for (std::size_t i=0; i<expected.size(); ++i) {
        EXPECT_EQ(results[i].weight, expected[i].weight);
    }
}

template<typename Stream>
struct result {
    timestep_range steps;
    std::unordered_map<unsigned, Stream> streams;
    std::unordered_map<unsigned, std::vector<std::vector<arb_deliverable_event_data>>> expected;
};

template<typename Stream>
result<Stream> single_step() {
    // events for 3 cells and 2 mechanisms and according targets
    //
    // target handles                            | events
    // ===================================================================
    // target  cell  div mech_id  mech_index lid | t=[0,1)
    // ===================================================================
    // 0       0     0   0        0          0   | e@t=0.0,w=0.0
    // -------------------------------------------------------------------
    // 1       0         1        0          1   |   e@t=0.1,w=1.0
    // ===================================================================
    // 2       1     2   0        0          0   |           e@t=0.5,w=0.2
    // 3       1         0        1          1   |   e@t=0.1,w=0.3
    // -------------------------------------------------------------------
    // 4       1         1        0          2   |         e@t=0.4,w=1.2
    // ===================================================================
    // 5       3     5   0        0          0   |         e@t=0.4,w=0.1
    // 6       3         0        1          1   |
    // -------------------------------------------------------------------
    // 7       3         1        0          2   |       e@t=0.3,w=1.1
    // 8       3         1        1          3   |
    // 9       3         1        2          4   |       e@t=0.3,w=1.3
    // ===================================================================
    //               10

    const std::vector<target_handle> handles = {
        {0, 0},
        {1, 0},
        {0, 0},
        {0, 1},
        {1, 0},
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1},
        {1, 2}
    };

    const std::vector<std::size_t> divs = {0, 2, 5, 10};

    std::vector<std::vector<spike_event>> events = {
        {{0, 0.0, 0.0f}, {1, 0.0, 1.0f}},
        {{1, 0.1, 0.3f}, {2, 0.4, 1.2f}, {0, 0.5, 0.2f}},
        {{2, 0.3, 1.1f}, {4, 0.3, 1.3f}, {0, 0.4, 0.1f}}
    };

    // prepare return value
    result<Stream> res {
        timestep_range{0,1,1},
        {{0u, Stream{}}, {1u, Stream{}}},
        {}
    };

    // expected outcome: one stream per mechanism, events ordered
    res.expected[0u] = std::vector<std::vector<arb_deliverable_event_data>>{
        { {0, 0.0f}, {0, 0.1f}, {0, 0.2f}, {1, 0.3f} } };
    res.expected[1u] = std::vector<std::vector<arb_deliverable_event_data>>{
        { {0, 1.0f}, {0, 1.1f}, {0, 1.2f}, {2, 1.3f} } };

    // initialize event streams
    auto lanes = util::subrange_view(events, 0u, events.size());
    initialize(lanes, handles, divs, res.steps, res.streams);

    return res;
}

template<typename Stream>
result<Stream> multi_step() {
    // number of events, cells, mechanisms and targets
    std::size_t num_events = 500;
    std::size_t num_cells = 20;
    std::size_t num_mechanisms = 5;
    std::size_t num_targets_per_mechanism_and_cell = 8;
    std::size_t num_steps = 200;
    double end_time = 1.0;

    result<Stream> res {
        timestep_range(0.0, end_time, end_time/num_steps),
        {},
        {}
    };
    for (std::size_t mech_id=0; mech_id<num_mechanisms; ++mech_id)
        res.streams[mech_id] = Stream{};

    // compute handles and divs
    std::vector<std::size_t> divs(num_cells+1, 0u);
    std::vector<target_handle> handles;
    handles.reserve(num_cells*num_mechanisms*num_targets_per_mechanism_and_cell);
    for (std::size_t cell=0; cell<num_cells; ++cell) {
        for (std::size_t mech_id=0; mech_id<num_mechanisms; ++mech_id) {
            for (std::size_t mech_index=0; mech_index<num_targets_per_mechanism_and_cell; ++mech_index) {
                handles.emplace_back(static_cast<cell_local_size_type>(mech_id), static_cast<cell_local_size_type>(mech_index));
            }
        }
        divs[cell+1] = divs[cell] + num_mechanisms*num_targets_per_mechanism_and_cell;
    }

    // events are binned by cell
    std::vector<std::vector<spike_event>> events(num_cells);

    // generate random events
    std::mt19937 gen(42);
    std::uniform_int_distribution<> cell_dist(0, num_cells-1);
    std::uniform_int_distribution<> mech_id_dist(0, num_mechanisms-1);
    std::uniform_int_distribution<> mech_index_dist(0, num_targets_per_mechanism_and_cell-1);
    std::uniform_real_distribution<> time_dist(0.0, end_time);
    for (std::size_t i=0; i<num_events; ++i) {
        auto cell = cell_dist(gen);
        auto mech_id = mech_id_dist(gen);
        auto mech_index = mech_index_dist(gen);
        auto target = mech_id*num_targets_per_mechanism_and_cell + mech_index;
        auto time = time_dist(gen);
        auto weight = 0.0f;
        events[cell].emplace_back(static_cast<cell_lid_type>(target), time, weight);
    }

    // sort events by time
    for (auto& v : events) {
        std::stable_sort(v.begin(), v.end(), [](auto const& l, auto const& r) { return l.time < r.time; });
    }

    // compute expected order as permutation of a pair which indexes into events:
    // first index is cell id, second index is item index
    std::vector<std::pair<std::size_t, std::size_t>> expected_order;
    expected_order.reserve(num_events);
    for (std::size_t cell=0; cell<num_cells; ++cell) {
        auto const& evts = events[cell];
        for (std::size_t i=0; i<evts.size(); ++i) {
            expected_order.push_back(std::make_pair(cell, i));
        }
    }
    std::sort(expected_order.begin(), expected_order.end(), [&](auto const& l, auto const& r) {
        auto [l_cell, l_idx] = l;
        auto [r_cell, r_idx] = r;
        auto const& l_event = events[l_cell][l_idx];
        auto const& r_event = events[r_cell][r_idx];
        auto l_t0 = res.steps.find(l_event.time)->t_begin();
        auto r_t0 = res.steps.find(r_event.time)->t_begin();
        auto const& l_handle = handles[divs[l_cell] + l_event.target];
        auto const& r_handle = handles[divs[r_cell] + r_event.target];
        auto l_mech_id = l_handle.mech_id;
        auto r_mech_id = r_handle.mech_id;
        auto l_mech_index = l_handle.mech_index;
        auto r_mech_index = r_handle.mech_index;

        // sort by mech_id
        if (l_mech_id < r_mech_id) return true;
        if (l_mech_id > r_mech_id) return false;
        // if same mech_id, sort by step
        if (l_t0 < r_t0) return true;
        if (l_t0 > r_t0) return false;
        // if same step, sort by mech_index
        if (l_mech_index < r_mech_index) return true;
        if (l_mech_index > r_mech_index) return false;
        // if same mech_index sort by time
        return l_event.time < r_event.time;
    });

    // expected results are now mapped by mechanism id -> vector of vector of deliverable event data
    // the outer vector represents time step bins, the inner vector the ordered stream of events
    for (std::size_t mech_id=0; mech_id<num_mechanisms; ++mech_id) {
        res.expected[mech_id] = std::vector<std::vector<arb_deliverable_event_data>>(num_steps);
    }

    // create expected results from the previously defined expected order and choose unique event weight
    std::size_t cc=0;
    for (auto [cell, idx] : expected_order) {
        auto& event = events[cell][idx];
        auto step = res.steps.find(event.time) - res.steps.begin();
        auto const& handle = handles[divs[cell] + event.target];
        auto mech_id = handle.mech_id;
        auto mech_index = handle.mech_index;
        event.weight = cc++;
        res.expected[mech_id][step].push_back(arb_deliverable_event_data{mech_index, event.weight});
    }

    // initialize event streams
    auto lanes = util::subrange_view(events, 0u, events.size());
    initialize(lanes, handles, divs, res.steps, res.streams);

    return res;
}

}
