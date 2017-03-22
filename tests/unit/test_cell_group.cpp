#include "../gtest.h"

#include <cell_group.hpp>
#include <common_types.hpp>
#include <fvm_multicell.hpp>
#include <util/rangeutil.hpp>

#include "common.hpp"
#include "../test_common_cells.hpp"

using namespace nest::mc;
using fvm_cell = fvm::fvm_multicell<nest::mc::multicore::backend>;

nest::mc::cell make_cell() {
    using namespace nest::mc;

    nest::mc::cell cell = make_cell_ball_and_stick();

    cell.add_detector({0, 0}, 0);
    cell.segment(1)->set_compartments(101);

    return cell;
}

TEST(cell_group, test) {
    cell_group<fvm_cell> group{0, util::singleton_view(make_cell())};

    group.advance(50, 0.01);

    // the model is expected to generate 4 spikes as a result of the
    // fixed stimulus over 50 ms
    EXPECT_EQ(4u, group.spikes().size());
}

TEST(cell_group, sources) {
    using cell_group_type = cell_group<fvm_cell>;

    auto cell = make_cell();
    EXPECT_EQ(cell.detectors().size(), 1u);
    // add another detector on the cell to make things more interesting
    cell.add_detector({1, 0.3}, 2.3);

    cell_gid_type first_gid = 37u;
    auto group = cell_group_type{first_gid, util::singleton_view(cell)};

    // expect group sources to be lexicographically sorted by source id
    // with gids in cell group's range and indices starting from zero

    const auto& sources = group.spike_sources();
    for (unsigned i = 0; i<sources.size(); ++i) {
        auto id = sources[i];
        if (i==0) {
            EXPECT_EQ(id.gid, first_gid);
            EXPECT_EQ(id.index, 0u);
        }
        else {
            auto prev = sources[i-1];
            EXPECT_GT(id, prev);
            EXPECT_EQ(id.index, id.gid==prev.gid? prev.index+1: 0u);
        }
    }
}

TEST(cell_group, event_binner) {
    using testing::seq_almost_eq;

    std::pair<cell_gid_type, float> binning_test_data[] = {
        { 11, 0.50 },
        { 12, 0.70 },
        { 14, 0.73 },
        { 11, 1.80 },
        { 12, 1.83 },
        { 11, 1.90 },
        { 11, 2.00 },
        { 14, 2.00 },
        { 11, 2.10 },
        { 14, 2.30 }
    };

    std::unordered_map<cell_gid_type, std::vector<float>> ev_times;
    std::vector<float> expected;

    auto run_binner = [&](event_binner&& binner) {
        ev_times.clear();
        for (auto p: binning_test_data) {
            ev_times[p.first].push_back(binner.bin(p.first, p.second));
        }
    };

    run_binner(event_binner{binning_kind::none, 0});

    EXPECT_TRUE(seq_almost_eq<float>(ev_times[11], (float []){0.50, 1.80, 1.90, 2.00, 2.10}));
    EXPECT_TRUE(seq_almost_eq<float>(ev_times[12], (float []){0.70, 1.83}));
    EXPECT_TRUE(ev_times[13].empty());
    EXPECT_TRUE(seq_almost_eq<float>(ev_times[14], (float []){0.73, 2.00, 2.30}));

    run_binner(event_binner{binning_kind::regular, 0.25});

    EXPECT_TRUE(seq_almost_eq<float>(ev_times[11], (float []){0.50, 1.75, 1.75, 2.00, 2.00}));
    EXPECT_TRUE(seq_almost_eq<float>(ev_times[12], (float []){0.50, 1.75}));
    EXPECT_TRUE(ev_times[13].empty());
    EXPECT_TRUE(seq_almost_eq<float>(ev_times[14], (float []){0.50, 2.00, 2.25}));

    run_binner(event_binner{binning_kind::following, 0.25});

    EXPECT_TRUE(seq_almost_eq<float>(ev_times[11], (float []){0.50, 1.80, 1.80, 1.80, 2.10}));
    EXPECT_TRUE(seq_almost_eq<float>(ev_times[12], (float []){0.70, 1.83}));
    EXPECT_TRUE(ev_times[13].empty());
    EXPECT_TRUE(seq_almost_eq<float>(ev_times[14], (float []){0.73, 2.00, 2.30}));
}

TEST(cell_group, event_binner_with_min) {
    using testing::seq_almost_eq;

    struct test_time {
        float time;
        float t_min;
    };
    test_time test_data[] = {
        {0.8f, 1.0f},
        {1.6f, 1.0f},
        {1.9f, 1.8f},
        {2.0f, 1.8f},
        {2.2f, 1.8f}
    };

    std::vector<float> times;
    auto run_binner = [&](event_binner&& binner, bool use_min) {
        times.clear();
        for (auto p: test_data) {
            if (use_min) {
                times.push_back(binner.bin(0, p.time, p.t_min));
            }
            else {
                times.push_back(binner.bin(0, p.time));
            }
        }
    };

    // 'none' binning
    run_binner(event_binner{binning_kind::none, 0.5}, false);
    EXPECT_TRUE(seq_almost_eq<float>(times, (float []){0.8, 1.6, 1.9, 2.0, 2.2}));

    run_binner(event_binner{binning_kind::none, 0.5}, true);
    EXPECT_TRUE(seq_almost_eq<float>(times, (float []){1.0, 1.6, 1.9, 2.0, 2.2}));

    // 'regular' binning
    run_binner(event_binner{binning_kind::regular, 0.5}, false);
    EXPECT_TRUE(seq_almost_eq<float>(times, (float []){0.5, 1.5, 1.5, 2.0, 2.0}));

    run_binner(event_binner{binning_kind::regular, 0.5}, true);
    EXPECT_TRUE(seq_almost_eq<float>(times, (float []){1.0, 1.5, 1.8, 2.0, 2.0}));

    // 'following' binning
    run_binner(event_binner{binning_kind::following, 0.5}, false);
    EXPECT_TRUE(seq_almost_eq<float>(times, (float []){0.8, 1.6, 1.6, 1.6, 2.2}));

    run_binner(event_binner{binning_kind::following, 0.5}, true);
    EXPECT_TRUE(seq_almost_eq<float>(times, (float []){1.0, 1.6, 1.8, 1.8, 2.2}));
}
