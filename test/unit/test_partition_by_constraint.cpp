#include "../gtest.h"

#include <array>
#include <forward_list>
#include <string>
#include <vector>

#include <arbor/common_types.hpp>
#include <arbor/mechanism_ppack.hpp>
#include <arbor/simd/simd.hpp>

#include "util/range.hpp"
#include "backends/multicore/multicore_common.hpp"
#include "backends/multicore/partition_by_constraint.hpp"

using namespace arb;
using iarray = multicore::iarray;
constexpr unsigned vector_length = (unsigned) simd::simd_abi::native_width<fvm_value_type>::value;
using simd_value_type = simd::simd<fvm_value_type, vector_length, simd::simd_abi::default_abi>;
const int simd_width_ = simd::width<simd_value_type>();

const int input_size_ = 1024;

iarray get_contiguous(const constraint_partition& output)  {
    auto v = util::range_n(output.contiguous, output.n_contiguous);
    return { v.begin(), v.end() };
}

iarray get_constant(const constraint_partition& output) {
    auto v = util::range_n(output.constant, output.n_constant);
    return { v.begin(), v.end() };
}

iarray get_independent(const constraint_partition& output) {
    auto v = util::range_n(output.independent, output.n_independent);
    return { v.begin(), v.end() };
}

iarray get_none(const constraint_partition& output) {
    auto v = util::range_n(output.none, output.n_none);
    return { v.begin(), v.end() };
}

TEST(partition_by_constraint, partition_contiguous) {
    iarray input_index(input_size_);
    iarray expected;

    for (unsigned i = 0; i < input_size_; i++) {
        input_index[i] = i;
        if(i % simd_width_ == 0)
            expected.push_back(i);
    }

    auto output = multicore::make_constraint_partition(input_index, input_size_, simd_width_);
    EXPECT_EQ(0u, output.n_independent);
    EXPECT_EQ(0u, output.n_none);
    EXPECT_EQ(0u, output.n_constant);
    EXPECT_EQ(expected, get_contiguous(output));
}

TEST(partition_by_constraint, partition_constant) {
    iarray input_index(input_size_);
    iarray expected;

    const int c = 5;

    for (unsigned i = 0; i < input_size_; i++) {
        input_index[i] = c;
        if(i % simd_width_ == 0)
            expected.push_back(i);
    }

    auto output = multicore::make_constraint_partition(input_index, input_size_, simd_width_);

    EXPECT_EQ(0u, output.n_independent);
    EXPECT_EQ(0u, output.n_none);
    if(simd_width_ != 1) {
        EXPECT_EQ(0u, output.n_contiguous);
        EXPECT_EQ(expected, get_constant(output));
    }
    else {
        EXPECT_EQ(0u, output.n_constant);
        EXPECT_EQ(expected, get_contiguous(output));
    }
}

TEST(partition_by_constraint, partition_independent) {
    iarray input_index(input_size_);
    iarray expected;
    constraint_partition output;

    for (unsigned i = 0; i < input_size_; i++) {
        input_index[i] = i * 2;
        if(i % simd_width_ == 0)
            expected.push_back(i);
    }

    output = multicore::make_constraint_partition(input_index, input_size_, simd_width_);

    EXPECT_EQ(0u, output.n_constant);
    EXPECT_EQ(0u, output.n_none);
    if(simd_width_ != 1) {
        EXPECT_EQ(0u, output.n_contiguous);
        EXPECT_EQ(expected, get_independent(output));
    }
    else {
        EXPECT_EQ(0u, output.n_independent);
        EXPECT_EQ(expected, get_contiguous(output));
    }
}

TEST(partition_by_constraint, partition_none) {
    iarray input_index(input_size_);
    iarray expected;
    constraint_partition output;

    for (unsigned i = 0; i < input_size_; i++) {
        input_index[i] = i / ((simd_width_ + 1)/ 2);
        if(i % simd_width_ == 0)
            expected.push_back(i);
    }

    output = multicore::make_constraint_partition(input_index, input_size_, simd_width_);

    EXPECT_EQ(0u, output.n_independent);
    EXPECT_EQ(0u, output.n_constant);
    if(simd_width_ > 2) {
        EXPECT_EQ(0u, output.n_contiguous);
        EXPECT_EQ(expected, get_none(output));
    }
    else {
        EXPECT_EQ(0u, output.n_none);
        EXPECT_EQ(expected, get_contiguous(output));
    }
}

TEST(partition_by_constraint, partition_random) {
    iarray input_index(input_size_);
    iarray expected_contiguous, expected_constant,
            expected_independent, expected_none, expected_simd_1;

    const int c = 5;
    for (unsigned i = 0; i < input_size_; i++) {
        input_index[i] = i<input_size_/4   ? i/((simd_width_ + 1)/ 2):
                         i<input_size_/2   ? i*2:
                         i<input_size_*3/4 ? c:
                         i;
        if (i < input_size_ / 4 && i % simd_width_ == 0) {
            if (simd_width_ > 2) {
                expected_none.push_back(i);
            } else {
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

    auto output = multicore::make_constraint_partition(input_index, input_size_, simd_width_);

    if (simd_width_ != 1) {
        EXPECT_EQ(expected_contiguous, get_contiguous(output));
        EXPECT_EQ(expected_constant, get_constant(output));
        EXPECT_EQ(expected_independent, get_independent(output));
        EXPECT_EQ(expected_none, get_none(output));
    }
    else {
        EXPECT_EQ(expected_simd_1, get_contiguous(output));
    }

}
