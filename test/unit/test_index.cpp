#include <forward_list>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "util/index_into.hpp"
#include "util/rangeutil.hpp"

#include "common.hpp"

template <typename Sub, typename Sup>
::testing::AssertionResult validate_index_into(const Sub& sub, const Sup& sup) {
    using namespace arb;

    auto indices = util::index_into(sub, sup);
    auto n_indices = std::size(indices);
    auto n_sub  = std::size(sub);
    if (std::size(indices)!=std::size(sub)) {
        return ::testing::AssertionFailure()
             << "index_into size " << n_indices << " does not equal sub-sequence size " << n_sub;
    }

    using std::begin;
    using std::end;

    auto sub_i = begin(sub);
    auto sup_i = begin(sup);
    auto sup_end = end(sup);
    std::ptrdiff_t sup_idx = 0;

    for (auto i: indices) {
        if (sup_idx>i) {
            return ::testing::AssertionFailure() << "indices in index_into sequence not monotonic";
        }

        while (sup_idx<i && sup_i!=sup_end) ++sup_idx, ++sup_i;

        if (sup_i==sup_end) {
            return ::testing::AssertionFailure() << "index " << i << "in index_into sequence is past the end";
        }

        if (!(*sub_i==*sup_i)) {
            return ::testing::AssertionFailure()
                << "value mismatch: sub-sequence element " << *sub_i
                << " not equal to super-sequence element " << *sup_i << " at index " << i;
        }

        ++sub_i;
    }

    return ::testing::AssertionSuccess();
}

TEST(util, index_into)
{
    using ivector = std::vector<std::ptrdiff_t>;
    using std::size;
    using arb::util::index_into;
    using arb::util::assign_from;
    using arb::util::make_range;
    using arb::util::all_of;

    std::vector<std::pair<std::vector<int>, std::vector<int>>> vector_tests = {
        // Empty sequences:
        {{}, {}},
        {{100}, {}},
        // Strictly monotonic sequences:
        {{0, 7}, {0, 7}},
        {{0, 1, 3, 4, 6, 7, 10, 11}, {0, 4, 6, 7, 11}},
        {{0, 1, 3, 4, 6, 7, 10, 11}, {0}},
        {{0, 1, 3, 4, 6, 7, 10, 11}, {11}},
        {{0, 1, 3, 4, 6, 7, 10, 11}, {4}},
        {{0, 1, 3, 4, 6, 7, 10, 11}, {0, 11}},
        {{0, 1, 3, 4, 6, 7, 10, 11}, {4, 11}},
        {{0, 1, 3, 4, 6, 7, 10, 11}, {0, 1, 3, 4, 6, 7, 10, 11}},
        // Sequences with duplicates:
        {{8, 8, 10, 10, 12, 12, 12, 13}, {8, 10, 13}},
        {{8, 8, 10, 10, 12, 12, 12, 13}, {10, 10, 13}},
        // Unordered sequences:
        {{10, 3, 7, -8, 11, -8, 1, 2}, {3, -8, -8, 1}}
    };

    for (auto& testcase: vector_tests) {
        EXPECT_TRUE(validate_index_into(testcase.second, testcase.first));
    }

    // Test across array types.

    int subarr1[] = {2, 3, 9, 5};
    int suparr1[] = {10, 2, 9, 3, 8, 5, 9, 3, 5, 10};
    EXPECT_TRUE(validate_index_into(subarr1, suparr1));

    // Test bidirectionality.

    auto arr_indices = index_into(subarr1, suparr1);
    ivector arridx = assign_from(arr_indices);

    ivector revidx;
    for (auto i = arr_indices.end(); i!=arr_indices.begin(); ) {
        revidx.push_back(*--i);
    }

    std::vector<std::ptrdiff_t> expected(arridx);
    std::reverse(expected.begin(), expected.end());
    EXPECT_EQ(expected, revidx);

    int subarr2[] = {8, 8, 8, 8, 8};
    int suparr2[] = {8};

    auto z_indices = index_into(subarr2, suparr2);
    EXPECT_TRUE(all_of(z_indices, [](std::ptrdiff_t n) { return n==0; }));
    EXPECT_EQ(0, z_indices.back());

    // Test: strictly forward sequences; heterogenous sequences; sentinel-terminated ranges.

    std::forward_list<double> sup_flist = {10.0, 2.1, 8.0, 3.8, 4.0, 4.0, 7.0, 1.0};
    std::forward_list<int> sub_flist = {8, 4, 4, 1};

    auto flist_indices = index_into(sub_flist, sup_flist);
    ivector idx_flist = assign_from(flist_indices);
    EXPECT_EQ((ivector{2, 4, 4, 7}), idx_flist);

    const char* hello_world = "hello world";
    const char* lol = "lol";
    auto sup_cstr = make_range(hello_world, testing::null_terminated);
    auto sub_cstr = make_range(lol, testing::null_terminated);
    auto cstr_indices = index_into(sub_cstr, sup_cstr);
    ivector idx_cstr = assign_from(cstr_indices);
    EXPECT_EQ((ivector{2, 4, 9}), idx_cstr);
}
