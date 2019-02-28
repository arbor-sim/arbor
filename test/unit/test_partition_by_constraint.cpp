#include "../gtest.h"

#include <array>
#include <forward_list>
#include <string>
#include <vector>

#include <arbor/common_types.hpp>
#include <arbor/simd/simd.hpp>

#include "backends/multicore/multicore_common.hpp"
#include "backends/multicore/partition_by_constraint.hpp"

using namespace arb;
using iarray = multicore::iarray;
static constexpr unsigned simd_width_ = arb::simd::simd_abi::native_width<fvm_value_type>::value;

const int input_size_ = 1024;

TEST(partition_by_constraint, partition_contiguous) {
    iarray input_index(input_size_);
    iarray expected;
    multicore::constraint_partition output;


    for (unsigned i = 0; i < input_size_; i++) {
        input_index[i] = i;
        if(i % simd_width_ == 0)
            expected.push_back(i);
    }

    output = multicore::make_constraint_partition(input_index, input_size_, simd_width_);

    EXPECT_EQ(0u, output.independent.size());
    EXPECT_EQ(0u, output.none.size());
    EXPECT_EQ(0u, output.constant.size());
    EXPECT_EQ(expected, output.contiguous);
}

TEST(partition_by_constraint, partition_constant) {
    iarray input_index(input_size_);
    iarray expected;
    multicore::constraint_partition output;

    const int c = 5;

    for (unsigned i = 0; i < input_size_; i++) {
        input_index[i] = c;
        if(i % simd_width_ == 0)
            expected.push_back(i);
    }

    output = multicore::make_constraint_partition(input_index, input_size_, simd_width_);

    EXPECT_EQ(0u, output.independent.size());
    EXPECT_EQ(0u, output.none.size());
    if(simd_width_ != 1) {
        EXPECT_EQ(0u, output.contiguous.size());
        EXPECT_EQ(expected, output.constant);
    }
    else {
        EXPECT_EQ(0u, output.constant.size());
        EXPECT_EQ(expected, output.contiguous);
    }
}

TEST(partition_by_constraint, partition_independent) {
    iarray input_index(input_size_);
    iarray expected;
    multicore::constraint_partition output;

    for (unsigned i = 0; i < input_size_; i++) {
        input_index[i] = i * 2;
        if(i % simd_width_ == 0)
            expected.push_back(i);
    }

    output = multicore::make_constraint_partition(input_index, input_size_, simd_width_);

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
    multicore::constraint_partition output;

    for (unsigned i = 0; i < input_size_; i++) {
        input_index[i] = i / ((simd_width_ + 1)/ 2);
        if(i % simd_width_ == 0)
            expected.push_back(i);
    }

    output = multicore::make_constraint_partition(input_index, input_size_, simd_width_);

    EXPECT_EQ(0u, output.independent.size());
    EXPECT_EQ(0u, output.constant.size());
    if(simd_width_ > 2) {
        EXPECT_EQ(0u, output.contiguous.size());
        EXPECT_EQ(expected, output.none);
    }
    else {
        EXPECT_EQ(0u, output.none.size());
        EXPECT_EQ(expected, output.contiguous);
    }
}

TEST(partition_by_constraint, partition_random) {
    iarray input_index(input_size_);
    iarray expected_contiguous, expected_constant,
            expected_independent, expected_none, expected_simd_1;
    multicore::constraint_partition output;


    const int c = 5;
    for (unsigned i = 0; i < input_size_; i++) {
        input_index[i] = i<input_size_/4   ? i/((simd_width_ + 1)/ 2):
                         i<input_size_/2   ? i*2:
                         i<input_size_*3/4 ? c:
                         i;
        if (i < input_size_ / 4 && i % simd_width_ == 0) {
            if (simd_width_ > 2) {
                expected_none.push_back(i);
            }
            else {
                expected_contiguous.push_back(i);
            }
        }
        else if (i < input_size_ / 2 && i % simd_width_ == 0)
            expected_independent.push_back(i);
        else if (i < input_size_* 3/ 4 && i % simd_width_ == 0)
            expected_constant.push_back(i);
        else if (i % simd_width_ == 0)
            expected_contiguous.push_back(i);
        expected_simd_1.push_back(i);

    }

    output = multicore::make_constraint_partition(input_index, input_size_, simd_width_);

    if (simd_width_ != 1) {
        EXPECT_EQ(expected_contiguous, output.contiguous);
        EXPECT_EQ(expected_constant, output.constant);
        EXPECT_EQ(expected_independent, output.independent);
        EXPECT_EQ(expected_none, output.none);
    }
    else {
        EXPECT_EQ(expected_simd_1, output.contiguous);
    }

}
