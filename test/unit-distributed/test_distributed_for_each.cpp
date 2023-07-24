#include <gtest/gtest.h>
#include "test.hpp"

#include <tuple>
#include <vector>
#include <complex>

#include "communication/distributed_for_each.hpp"
#include "execution_context.hpp"
#include "util/range.hpp"

using namespace arb;

// check when all input is size 0
TEST(distributed_for_each, all_zero) {
    std::vector<int> data;

    const int num_ranks = g_context->distributed->size();
    int call_count = 0;

    auto sample = [&](const util::range<int*>& range) {
        EXPECT_EQ(0, range.size());
        ++call_count;
    };

    distributed_for_each(
        sample, *g_context->distributed, util::make_range(data.begin(), data.end()));

    EXPECT_EQ(num_ranks, call_count);
}

// check when input on one rank is size 0
TEST(distributed_for_each, one_zero) {
    const auto rank = g_context->distributed->id();
    const int num_ranks = g_context->distributed->size();
    int call_count = 0;

    // test data size is equal to rank id and vector is filled with rank id
    std::vector<int> data;
    for (int i = 0; i < rank; ++i) { data.push_back(rank); }

    auto sample = [&](const util::range<int*>& range) {
        const auto origin_rank = range.empty() ? 0 : range.front();

        EXPECT_EQ(origin_rank, range.size());
        for (const auto& value: range) { EXPECT_EQ(value, origin_rank); }
        ++call_count;
    };

    distributed_for_each(
        sample, *g_context->distributed, util::make_range(data.begin(), data.end()));

    EXPECT_EQ(num_ranks, call_count);
}

// check multiple types
TEST(distributed_for_each, multiple) {
    const auto rank = g_context->distributed->id();
    const int num_ranks = g_context->distributed->size();
    int call_count = 0;

    std::vector<int> data_1;
    std::vector<double> data_2;
    std::vector<std::complex<double>> data_3;
    // test data size is equal to rank id + 1and vector is filled with rank id
    for (int i = 0; i < rank + 1; ++i) { data_1.push_back(rank); }
    // test different data sizes for each type
    for (std::size_t i = 0; i < 2 * data_1.size(); ++i) { data_2.push_back(rank); }
    for (std::size_t i = 0; i < 3 * data_1.size(); ++i) { data_3.push_back(rank); }

    auto sample = [&](const util::range<int*>& range_1,
                      const util::range<double*>& range_2,
                      const util::range<std::complex<double>*>& range_3) {
        const auto origin_rank = range_1.empty() ? 0 : range_1.front();

        EXPECT_EQ(origin_rank + 1, range_1.size());
        EXPECT_EQ(range_2.size(), 2 * range_1.size());
        EXPECT_EQ(range_3.size(), 3 * range_1.size());
        for (const auto& value: range_1) { EXPECT_EQ(value, origin_rank); }
        for (const auto& value: range_2) { EXPECT_EQ(value, double(origin_rank)); }
        for (const auto& value: range_3) { EXPECT_EQ(value, std::complex<double>(origin_rank)); }
        ++call_count;
    };

    distributed_for_each(sample,
        *g_context->distributed,
        util::make_range(data_1.begin(), data_1.end()),
        util::make_range(data_2.begin(), data_2.end()),
        util::make_range(data_3.begin(), data_3.end()));

    EXPECT_EQ(num_ranks, call_count);
}
