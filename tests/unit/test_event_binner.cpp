#include "../gtest.h"

#include <event_binner.hpp>

#include "common.hpp"

using namespace nest::mc;

TEST(event_binner, basic) {
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

TEST(event_binner, with_min) {
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
