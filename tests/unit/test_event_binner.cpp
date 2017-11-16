#include "../gtest.h"

#include <event_binner.hpp>

#include "common.hpp"

using namespace arb;

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
                times.push_back(binner.bin(p.time, p.t_min));
            }
            else {
                times.push_back(binner.bin(p.time));
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
