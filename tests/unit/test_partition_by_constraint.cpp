#include "../gtest.h"

#include <array>
#include <forward_list>
#include <string>
#include <vector>

#include <simd/simd.hpp>
#include <common_types.hpp>
#include <backends/multicore/multicore_common.hpp>
#include <backends/multicore/partition_by_constraint.hpp>

using namespace arb;
using iarray = multicore::iarray;
static constexpr unsigned simd_width_ = arb::simd::simd_abi::native_width<fvm_value_type>::value;

const int input_size_ = 1024;

TEST(partition_by_constraint, partition_contiguous) {
    iarray input_index(input_size_);
    iarray expected_indices;
    multicore::constraint_partitions output_constraint;


    for (unsigned i = 0; i < input_size_; i++) {
        input_index[i] = i;
        if(i % simd_width_ == 0)
            expected_indices.push_back(i);
    }

    multicore::gen_constraint(input_index, output_constraint, input_size_);

    EXPECT_EQ(input_size_/simd_width_, output_constraint.contiguous_indices.size());
    EXPECT_EQ(0, output_constraint.constant_indices.size());
    EXPECT_EQ(0, output_constraint.independent_indices.size());
    EXPECT_EQ(0, output_constraint.serialized_indices.size());

    for (unsigned i = 0; i < input_size_ / simd_width_; i++) {
        EXPECT_EQ(expected_indices[i], output_constraint.contiguous_indices[i]);
    }
}

TEST(partition_by_constraint, partition_constant) {
    iarray input_index(input_size_);
    iarray expected_indices;
    multicore::constraint_partitions output_constraint;

    const int c = 5;

    for (unsigned i = 0; i < input_size_; i++) {
        input_index[i] = c;
        if(i % simd_width_ == 0)
            expected_indices.push_back(i);
    }

    multicore::gen_constraint(input_index, output_constraint, input_size_);

    if(simd_width_ != 1) {
        EXPECT_EQ(input_size_/simd_width_, output_constraint.constant_indices.size());
        EXPECT_EQ(0, output_constraint.contiguous_indices.size());
    }
    else {
        EXPECT_EQ(0, output_constraint.constant_indices.size());
        EXPECT_EQ(input_size_/simd_width_, output_constraint.contiguous_indices.size());
    }
    EXPECT_EQ(0, output_constraint.independent_indices.size());
    EXPECT_EQ(0, output_constraint.serialized_indices.size());

    for (unsigned i = 0; i < input_size_ / simd_width_; i++) {
        if(simd_width_ != 1)
            EXPECT_EQ(expected_indices[i], output_constraint.constant_indices[i]);
        else
            EXPECT_EQ(expected_indices[i], output_constraint.contiguous_indices[i]);
    }
}

TEST(partition_by_constraint, partition_independent) {
    iarray input_index(input_size_);
    iarray expected_indices;
    multicore::constraint_partitions output_constraint;

    for (unsigned i = 0; i < input_size_; i++) {
        input_index[i] = i * 2;
        if(i % simd_width_ == 0)
            expected_indices.push_back(i);
    }

    multicore::gen_constraint(input_index, output_constraint, input_size_);

    if(simd_width_ != 1) {
        EXPECT_EQ(input_size_/simd_width_, output_constraint.independent_indices.size());
        EXPECT_EQ(0, output_constraint.contiguous_indices.size());
    }
    else {
        EXPECT_EQ(0, output_constraint.independent_indices.size());
        EXPECT_EQ(input_size_/simd_width_, output_constraint.contiguous_indices.size());
    }
    EXPECT_EQ(0, output_constraint.constant_indices.size());
    EXPECT_EQ(0, output_constraint.serialized_indices.size());

    for (unsigned i = 0; i < input_size_ / simd_width_; i++) {
        if(simd_width_ != 1)
            EXPECT_EQ(expected_indices[i], output_constraint.independent_indices[i]);
        else
            EXPECT_EQ(expected_indices[i], output_constraint.contiguous_indices[i]);
    }
}

TEST(partition_by_constraint, partition_serial) {
    iarray input_index(input_size_);
    iarray expected_indices;
    multicore::constraint_partitions output_constraint;

    for (unsigned i = 0; i < input_size_; i++) {
        input_index[i] = i / ((simd_width_ + 1)/ 2);
        if(i % simd_width_ == 0)
            expected_indices.push_back(i);
    }

    multicore::gen_constraint(input_index, output_constraint, input_size_);

    if(simd_width_ != 1) {
        EXPECT_EQ(input_size_/simd_width_, output_constraint.serialized_indices.size());
        EXPECT_EQ(0, output_constraint.contiguous_indices.size());
    }
    else {
        EXPECT_EQ(0, output_constraint.serialized_indices.size());
        EXPECT_EQ(input_size_/simd_width_, output_constraint.contiguous_indices.size());
    }
    EXPECT_EQ(0, output_constraint.independent_indices.size());
    EXPECT_EQ(0, output_constraint.constant_indices.size());

    for (unsigned i = 0; i < input_size_ / simd_width_; i++) {
        if(simd_width_ != 1)
            EXPECT_EQ(expected_indices[i], output_constraint.serialized_indices[i]);
        else
            EXPECT_EQ(expected_indices[i], output_constraint.contiguous_indices[i]);
    }
}

TEST(partition_by_constraint, partition_random) {
    iarray input_index(input_size_);
    iarray expected_indices_contiguous, expected_indices_constant,
            expected_indices_independent, expected_indices_serial;
    multicore::constraint_partitions output_constraint;


    const int c = 5;
//shuffle here
    for (unsigned i = 0; i < input_size_; i++) {
        input_index[i] = (i < input_size_ / 4 ? i / ((simd_width_ + 1)/ 2) :
                          (i < input_size_ / 2 ? i * 2 :
                           (i < input_size_* 3 / 4 ? c : i)));
        if (i < input_size_ / 4 && i % simd_width_ == 0)
            expected_indices_serial.push_back(i);
        else if (i < input_size_ / 2 && i % simd_width_ == 0)
            expected_indices_independent.push_back(i);
        else if (i < input_size_* 3/ 4 && i % simd_width_ == 0)
            expected_indices_constant.push_back(i);
        else if (i % simd_width_ == 0)
            expected_indices_contiguous.push_back(i);
    }

    multicore::gen_constraint(input_index, output_constraint, input_size_);

    if (simd_width_ != 1) {
        EXPECT_EQ(input_size_ / 4 / simd_width_, output_constraint.contiguous_indices.size());
        EXPECT_EQ(input_size_ / 4 / simd_width_, output_constraint.constant_indices.size());
        EXPECT_EQ(input_size_ / 4 / simd_width_, output_constraint.independent_indices.size());
        EXPECT_EQ(input_size_ / 4 / simd_width_, output_constraint.serialized_indices.size());
    }
    else {
        EXPECT_EQ(input_size_ / simd_width_, output_constraint.contiguous_indices.size());
        EXPECT_EQ(0, output_constraint.constant_indices.size());
        EXPECT_EQ(0, output_constraint.independent_indices.size());
        EXPECT_EQ(0, output_constraint.serialized_indices.size());
    }

    if (simd_width_ != 1) {
        for (unsigned i = 0; i < input_size_ / simd_width_ / 4; i++) {
            EXPECT_EQ(expected_indices_contiguous[i], output_constraint.contiguous_indices[i]);
            EXPECT_EQ(expected_indices_constant[i], output_constraint.constant_indices[i]);
            EXPECT_EQ(expected_indices_independent[i], output_constraint.independent_indices[i]);
            EXPECT_EQ(expected_indices_serial[i], output_constraint.serialized_indices[i]);
        }
    }
    else {
        for (unsigned i = 0; i < input_size_ / simd_width_; i++) {
            EXPECT_EQ(i, output_constraint.contiguous_indices[i]);
        }
    }
}

TEST(partition_by_constraint, partition_subvector_constraint_categorization) {
    iarray input_index(input_size_);
    multicore::constraint_partitions output_constraint;

    if(simd_width_ == 4) {
        {
            iarray input_index_constant0;
            for (unsigned i = 0; i < simd_width_; i++) {
                input_index_constant0.push_back(0);
            }
            EXPECT_EQ(multicore::get_subvector_index_constraint(input_index_constant0, 0), index_constraint::constant);
        }

        {
            iarray input_index_contiguous0;
            for (unsigned i = 0; i < simd_width_; i++) {
                input_index_contiguous0.push_back(i);
            }
            EXPECT_EQ(multicore::get_subvector_index_constraint(input_index_contiguous0, 0),
                      index_constraint::contiguous);
        }

        {
            iarray input_index_independent0;
            for (unsigned i = 0; i < simd_width_; i++) {
                input_index_independent0.push_back(i * 2);
            }
            EXPECT_EQ(multicore::get_subvector_index_constraint(input_index_independent0, 0),
                      index_constraint::independent);
        }

        {
            iarray input_index_serial0;
            for (unsigned i = 0; i < simd_width_; i++) {
                input_index_serial0.push_back(i / 2);
            }
            EXPECT_EQ(multicore::get_subvector_index_constraint(input_index_serial0, 0), index_constraint::none);
        }

        {
            iarray input_index_serial1 = iarray{1, 2, 2, 4};
            EXPECT_EQ(multicore::get_subvector_index_constraint(input_index_serial1, 0), index_constraint::none);
        }


        {
            iarray input_index_independent1 = iarray{0, 2, 656, 900};
            EXPECT_EQ(multicore::get_subvector_index_constraint(input_index_independent1, 0),
                      index_constraint::independent);

        }
    }
}
