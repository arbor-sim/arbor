#include <gtest/gtest.h>

#include <arbor/common_types.hpp>
#include <arbor/simd/simd.hpp>

#include "backends/multicore/multicore_common.hpp"
#include "backends/multicore/partition_by_constraint.hpp"

using namespace arb;
using iarray = multicore::iarray;
const int simd_width_ = 2;
const int input_size_ = 1024;

TEST(partition_by_constraint, partition_contiguous) {
    iarray input_index(input_size_);
    std::iota(input_index.begin(), input_index.end(), 0);

    auto output = multicore::make_constraint_partition(input_index, input_size_, simd_width_);
    iarray expected{0, input_size_};

    EXPECT_EQ(0u, output.independent.size());
    EXPECT_EQ(0u, output.none.size());
    EXPECT_EQ(0u, output.constant.size());
    EXPECT_EQ(expected, output.contiguous);
}

TEST(partition_by_constraint, partition_constant) {
    iarray input_index(input_size_);
    const int c = 5;
    std::fill(input_index.begin(), input_index.end(), c);

    iarray expected;
    for (unsigned i = 0; i < input_size_; i += simd_width_) {
        expected.push_back(i);
    }

    auto output = multicore::make_constraint_partition(input_index, input_size_, simd_width_);

    EXPECT_EQ(0u, output.independent.size());
    EXPECT_EQ(0u, output.none.size());
    if(simd_width_ != 1) {
        EXPECT_EQ(0u, output.contiguous.size());
        EXPECT_EQ(expected, output.constant);
    }
    else {
        iarray expected{0, input_size_};
        EXPECT_EQ(0u, output.constant.size());
        EXPECT_EQ(expected, expected);
    }
}

TEST(partition_by_constraint, partition_independent) {
    iarray input_index(input_size_);
    iarray expected;

    for (unsigned i = 0; i < input_size_; i++) {
        input_index[i] = i * 2;
        if(i % simd_width_ == 0) expected.push_back(i);
    }

    auto output = multicore::make_constraint_partition(input_index, input_size_, simd_width_);

    EXPECT_EQ(0u, output.constant.size());
    EXPECT_EQ(0u, output.none.size());
    if(simd_width_ != 1) {
        EXPECT_EQ(0u, output.contiguous.size());
        EXPECT_EQ(expected, output.independent);
    }
    else {
        EXPECT_EQ(0u, output.independent.size());
        EXPECT_EQ(expected, output.contiguous);
    }
}

TEST(partition_by_constraint, partition_none) {
    iarray input_index(input_size_);
    iarray expected;

    for (unsigned i = 0; i < input_size_; i++) {
        input_index[i] = i / ((simd_width_ + 1)/ 2);
        if(i % simd_width_ == 0) expected.push_back(i);
    }

    auto output = multicore::make_constraint_partition(input_index, input_size_, simd_width_);

    EXPECT_EQ(0u, output.independent.size());
    EXPECT_EQ(0u, output.constant.size());
    if(simd_width_ > 2) {
        EXPECT_EQ(0u, output.contiguous.size());
        EXPECT_EQ(expected, output.none);
    }
    else {
        EXPECT_EQ(0u, output.none.size());
        iarray expected{0, 1024};
        EXPECT_EQ(expected, output.contiguous);
    }
}

TEST(partition_by_constraint, partition_random) {
    auto check = [] (std::size_t simd_width, std::size_t input_size) {
        // process input in quarters
        // 1. no constraint
        // 2. independent
        // 3. constant
        // 4. contiguous
        iarray input_index(input_size);
        for (unsigned i = 0;              i <   input_size/4; i++) input_index[i] = i/((simd_width + 1)/ 2);
        for (unsigned i =   input_size/4; i <   input_size/2; i++) input_index[i] = i*2;
        for (unsigned i =   input_size/2; i < 3*input_size/4; i++) input_index[i] = 5;
        for (unsigned i = 3*input_size/4; i <   input_size;   i++) input_index[i] = i;

        iarray expected_constant, expected_independent, expected_none, expected_contiguous;
        for (unsigned i = 0;              i <   input_size/4; i += simd_width) expected_none.push_back(i);
        for (unsigned i =   input_size/2; i < 3*input_size/4; i += simd_width) expected_constant.push_back(i);
        for (unsigned i =   input_size/4; i <   input_size/2; i += simd_width) expected_independent.push_back(i);
        expected_contiguous = {3*static_cast<int>(input_size)/4, static_cast<int>(input_size)};
        // NOTE iff width == 2, the first 1/4 becomes either constant,
        //      contiguous or independent instead of none.
        //      Out choice here is contiguous.
        if (simd_width == 2) {
            expected_contiguous = {0, static_cast<int>(input_size)/4, 3*static_cast<int>(input_size)/4, static_cast<int>(input_size)};
            expected_none       = {};
        }

        auto output = multicore::make_constraint_partition(input_index, input_size, simd_width);
        EXPECT_EQ(expected_contiguous, output.contiguous);
        EXPECT_EQ(expected_constant, output.constant);
        EXPECT_EQ(expected_independent, output.independent);
        EXPECT_EQ(expected_none, output.none);
    };
    check(2, 128);
    check(4, 2048);
    check(8, 1024);
}
