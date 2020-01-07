#include <forward_list>
#include <iterator>
#include <random>
#include <string>
#include <vector>

#include "../gtest.h"

#include "algorithms.hpp"
#include "util/index_into.hpp"
#include "util/meta.hpp"
#include "util/rangeutil.hpp"

// (Pending abstraction of threading interface)
#include <arbor/version.hpp>
#include "threading/threading.hpp"
#include "common.hpp"

TEST(algorithms, make_index)
{
    {
        std::vector<int> v(10, 1);
        auto index = arb::algorithms::make_index(v);

        EXPECT_EQ(index.size(), 11u);
        EXPECT_EQ(index.front(), 0);
        EXPECT_EQ(index.back(), arb::util::sum(v));
    }

    {
        std::vector<int> v(10);
        std::iota(v.begin(), v.end(), 1);
        auto index = arb::algorithms::make_index(v);

        EXPECT_EQ(index.size(), 11u);
        EXPECT_EQ(index.front(), 0);
        EXPECT_EQ(index.back(), arb::util::sum(v));
    }
}

TEST(algorithms, minimal_degree)
{
    {
        std::vector<int> v = {0};
        EXPECT_TRUE(arb::algorithms::is_minimal_degree(v));
    }

    {
        std::vector<int> v = {0, 0, 1, 2, 3, 4};
        EXPECT_TRUE(arb::algorithms::is_minimal_degree(v));
    }

    {
        std::vector<int> v = {0, 0, 1, 2, 0, 4};
        EXPECT_TRUE(arb::algorithms::is_minimal_degree(v));
    }

    {
        std::vector<int> v = {0, 0, 1, 2, 0, 4, 5, 4};
        EXPECT_TRUE(arb::algorithms::is_minimal_degree(v));
    }

    {
        std::vector<int> v = {1};
        EXPECT_FALSE(arb::algorithms::is_minimal_degree(v));
    }

    {
        std::vector<int> v = {0, 2};
        EXPECT_FALSE(arb::algorithms::is_minimal_degree(v));
    }

    {
        std::vector<int> v = {0, 1, 2};
        EXPECT_FALSE(arb::algorithms::is_minimal_degree(v));
    }
}

TEST(algorithms, is_strictly_monotonic_increasing)
{
    EXPECT_TRUE(
        arb::algorithms::is_strictly_monotonic_increasing(
            std::vector<int>{0}
        )
    );
    EXPECT_TRUE(
        arb::algorithms::is_strictly_monotonic_increasing(
            std::vector<int>{0, 1, 2, 3}
        )
    );
    EXPECT_TRUE(
        arb::algorithms::is_strictly_monotonic_increasing(
            std::vector<int>{8, 20, 42, 89}
        )
    );
    EXPECT_FALSE(
        arb::algorithms::is_strictly_monotonic_increasing(
            std::vector<int>{0, 0}
        )
    );
    EXPECT_FALSE(
        arb::algorithms::is_strictly_monotonic_increasing(
            std::vector<int>{8, 20, 20, 89}
        )
    );
    EXPECT_FALSE(
        arb::algorithms::is_strictly_monotonic_increasing(
            std::vector<int>{3, 2, 1, 0}
        )
    );
}

TEST(algorithms, is_strictly_monotonic_decreasing)
{
    EXPECT_TRUE(
        arb::algorithms::is_strictly_monotonic_decreasing(
            std::vector<int>{0}
        )
    );
    EXPECT_TRUE(
        arb::algorithms::is_strictly_monotonic_decreasing(
            std::vector<int>{3, 2, 1, 0}
        )
    );
    EXPECT_FALSE(
        arb::algorithms::is_strictly_monotonic_decreasing(
            std::vector<int>{0, 1, 2, 3}
        )
    );
    EXPECT_FALSE(
        arb::algorithms::is_strictly_monotonic_decreasing(
            std::vector<int>{8, 20, 42, 89}
        )
    );
    EXPECT_FALSE(
        arb::algorithms::is_strictly_monotonic_decreasing(
            std::vector<int>{0, 0}
        )
    );
    EXPECT_FALSE(
        arb::algorithms::is_strictly_monotonic_decreasing(
            std::vector<int>{8, 20, 20, 89}
        )
    );
}

TEST(algorithms, all_positive) {
    using arb::algorithms::all_positive;

    EXPECT_TRUE(all_positive(std::vector<int>{}));
    EXPECT_TRUE(all_positive(std::vector<int>{3, 2, 1}));
    EXPECT_FALSE(all_positive(std::vector<int>{3, 2, 1, 0}));
    EXPECT_FALSE(all_positive(std::vector<int>{-1}));

    EXPECT_TRUE(all_positive((double []){1., 2.}));
    EXPECT_FALSE(all_positive((double []){1., 0.}));
    EXPECT_FALSE(all_positive((double []){NAN}));

    EXPECT_TRUE(all_positive((std::string []){"a", "b"}));
    EXPECT_FALSE(all_positive((std::string []){"a", "", "b"}));
}

TEST(algorithms, all_negative) {
    using arb::algorithms::all_negative;

    EXPECT_TRUE(all_negative(std::vector<int>{}));
    EXPECT_TRUE(all_negative(std::vector<int>{-3, -2, -1}));
    EXPECT_FALSE(all_negative(std::vector<int>{-3, -2, -1, 0}));
    EXPECT_FALSE(all_negative(std::vector<int>{1}));

    double negzero = std::copysign(0., -1.);

    EXPECT_TRUE(all_negative((double []){-1., -2.}));
    EXPECT_FALSE(all_negative((double []){-1., 0.}));
    EXPECT_FALSE(all_negative((double []){-1., negzero}));
    EXPECT_FALSE(all_negative((double []){NAN}));

    EXPECT_FALSE(all_negative((std::string []){"", "b"}));
    EXPECT_FALSE(all_negative((std::string []){""}));
}

TEST(algorithms, has_contiguous_compartments)
{
    //
    //       0
    //       |
    //       1
    //       |
    //       2
    //      /|\.
    //     3 7 4
    //    /     \.
    //   5       6
    //
    EXPECT_FALSE(
        arb::algorithms::has_contiguous_compartments(
            std::vector<int>{0, 0, 1, 2, 2, 3, 4, 2}
        )
    );

    //
    //       0
    //       |
    //       1
    //       |
    //       2
    //      /|\.
    //     3 6 5
    //    /     \.
    //   4       7
    //
    EXPECT_FALSE(
        arb::algorithms::has_contiguous_compartments(
            std::vector<int>{0, 0, 1, 2, 3, 2, 2, 5}
        )
    );

    //
    //       0
    //       |
    //       1
    //       |
    //       2
    //      /|\.
    //     3 7 5
    //    /     \.
    //   4       6
    //
    EXPECT_TRUE(
        arb::algorithms::has_contiguous_compartments(
            std::vector<int>{0, 0, 1, 2, 3, 2, 5, 2}
        )
    );

    //
    //         0
    //         |
    //         1
    //        / \.
    //       2   7
    //      / \.
    //     3   5
    //    /     \.
    //   4       6
    //
    EXPECT_TRUE(
        arb::algorithms::has_contiguous_compartments(
            std::vector<int>{0, 0, 1, 2, 3, 2, 5, 1}
        )
    );

    //
    //     0
    //    / \.
    //   1   2
    //  / \.
    // 3   4
    //
    EXPECT_TRUE(
        arb::algorithms::has_contiguous_compartments(
            std::vector<int>{0, 0, 0, 1, 1}
        )
    );

    // Soma-only list
    EXPECT_TRUE(
        arb::algorithms::has_contiguous_compartments(
            std::vector<int>{0}
        )
    );

    // Empty list
    EXPECT_TRUE(
        arb::algorithms::has_contiguous_compartments(
            std::vector<int>{}
        )
    );
}

TEST(algorithms, is_unique)
{
    EXPECT_TRUE(
        arb::algorithms::is_unique(
            std::vector<int>{}
        )
    );
    EXPECT_TRUE(
        arb::algorithms::is_unique(
            std::vector<int>{0}
        )
    );
    EXPECT_TRUE(
        arb::algorithms::is_unique(
            std::vector<int>{0,1,100}
        )
    );
    EXPECT_FALSE(
        arb::algorithms::is_unique(
            std::vector<int>{0,0}
        )
    );
    EXPECT_FALSE(
        arb::algorithms::is_unique(
            std::vector<int>{0,1,2,2,3,4}
        )
    );
    EXPECT_FALSE(
        arb::algorithms::is_unique(
            std::vector<int>{0,1,2,3,4,4}
        )
    );
}

TEST(algorithms, child_count)
{
    {
        //
        //        0
        //       /|\.
        //      1 4 6
        //     /  |  \.
        //    2   5   7
        //   /         \.
        //  3           8
        //             / \.
        //            9   11
        //           /     \.
        //          10      12
        //                   \.
        //                    13
        //
        std::vector<int> parent_index =
            { 0, 0, 1, 2, 0, 4, 0, 6, 7, 8, 9, 8, 11, 12 };
        std::vector<int> expected_child_count =
            { 3, 1, 1, 0, 1, 0, 1, 1, 2, 1, 0, 1, 1, 0 };

        // auto count = arb::algorithms::child_count(parent_index);
        EXPECT_EQ(expected_child_count,
                  arb::algorithms::child_count(parent_index));
    }

}

TEST(algorithms, branches)
{
    using namespace arb;

    {
        //
        //        0                        0
        //       /|\.                     /|\.
        //      1 4 6                    1 2 3
        //     /  |  \.           =>        / \.
        //    2   5   7                    4   5
        //   /         \.
        //  3           8
        //             / \.
        //            9   11
        //           /     \.
        //          10      12
        //                   \.
        //                    13
        //
        std::vector<int> parent_index =
            { 0, 0, 1, 2, 0, 4, 0, 6, 7, 8, 9, 8, 11, 12 };
        std::vector<int> expected_branches =
            { 0, 1, 4, 6, 9, 11, 14 };
        std::vector<int> expected_parent_index =
            { 0, 0, 0, 0, 3, 3 };

        auto actual_branches = algorithms::branches(parent_index);
        EXPECT_EQ(expected_branches, actual_branches);

        auto actual_parent_index =
            algorithms::tree_reduce(parent_index, actual_branches);
        EXPECT_EQ(expected_parent_index, actual_parent_index);

        // Check find_branch
        EXPECT_EQ(0, algorithms::find_branch(actual_branches,  0));
        EXPECT_EQ(1, algorithms::find_branch(actual_branches,  1));
        EXPECT_EQ(1, algorithms::find_branch(actual_branches,  2));
        EXPECT_EQ(1, algorithms::find_branch(actual_branches,  3));
        EXPECT_EQ(2, algorithms::find_branch(actual_branches,  4));
        EXPECT_EQ(2, algorithms::find_branch(actual_branches,  5));
        EXPECT_EQ(3, algorithms::find_branch(actual_branches,  6));
        EXPECT_EQ(3, algorithms::find_branch(actual_branches,  7));
        EXPECT_EQ(3, algorithms::find_branch(actual_branches,  8));
        EXPECT_EQ(4, algorithms::find_branch(actual_branches,  9));
        EXPECT_EQ(4, algorithms::find_branch(actual_branches, 10));
        EXPECT_EQ(5, algorithms::find_branch(actual_branches, 11));
        EXPECT_EQ(5, algorithms::find_branch(actual_branches, 12));
        EXPECT_EQ(5, algorithms::find_branch(actual_branches, 13));
        EXPECT_EQ(6, algorithms::find_branch(actual_branches, 55));

        // Check expand_branches
        auto expanded = algorithms::expand_branches(actual_branches);
        EXPECT_EQ(parent_index.size(), expanded.size());
        for (std::size_t i = 0; i < parent_index.size(); ++i) {
            EXPECT_EQ(algorithms::find_branch(actual_branches, i),
                      expanded[i]);
        }
    }

    {
        //
        //    0      0
        //    |      |
        //    1  =>  1
        //    |
        //    2
        //    |
        //    3
        //
        std::vector<int> parent_index          = { 0, 0, 1, 2 };
        std::vector<int> expected_branches     = { 0, 1, 4 };
        std::vector<int> expected_parent_index = { 0, 0 };

        auto actual_branches = algorithms::branches(parent_index);
        EXPECT_EQ(expected_branches, actual_branches);

        auto actual_parent_index =
            algorithms::tree_reduce(parent_index, actual_branches);
        EXPECT_EQ(expected_parent_index, actual_parent_index);
    }

    {
        //
        //    0           0
        //    |           |
        //    1     =>    1
        //    |          / \.
        //    2         2   3
        //   / \.
        //  3   4
        //       \.
        //        5
        //
        std::vector<int> parent_index          = { 0, 0, 1, 2, 2, 4 };
        std::vector<int> expected_branches     = { 0, 1, 3, 4, 6 };
        std::vector<int> expected_parent_index = { 0, 0, 1, 1 };

        auto actual_branches = algorithms::branches(parent_index);
        EXPECT_EQ(expected_branches, actual_branches);

        auto actual_parent_index =
            algorithms::tree_reduce(parent_index, actual_branches);
        EXPECT_EQ(expected_parent_index, actual_parent_index);
    }

    {
        std::vector<int> parent_index          = { 0 };
        std::vector<int> expected_branches     = { 0, 1 };
        std::vector<int> expected_parent_index = { 0 };

        auto actual_branches = algorithms::branches(parent_index);
        EXPECT_EQ(expected_branches, actual_branches);

        auto actual_parent_index =
            algorithms::tree_reduce(parent_index, actual_branches);
        EXPECT_EQ(expected_parent_index, actual_parent_index);
    }
}

struct test_index_into {
    template <typename R1, typename R2, typename R3>
    bool operator() (const R1& sub, const R2& super, const R3& index) {
        using value_type = typename R1::value_type;

        if(sub.size()!=index.size()) return false;
        auto index_it = index.begin();
        for(auto i=0u; i<sub.size(); ++i, ++index_it) {
            auto idx = *index_it;
            if(idx>=value_type(super.size())) return false;
            if(super[idx]!=sub[i]) return false;
        }

        return true;
    }
};

template <typename Sub, typename Sup>
::testing::AssertionResult validate_index_into(const Sub& sub, const Sup& sup) {
    using namespace arb;

    auto indices = util::index_into(sub, sup);
    auto n_indices = util::size(indices);
    auto n_sub  = util::size(sub);
    if (util::size(indices)!=util::size(sub)) {
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

template <typename I>
arb::util::range<std::reverse_iterator<I>> reverse_range(arb::util::range<I> r) {
    using reviter = std::reverse_iterator<I>;
    return arb::util::make_range(reviter(r.end()), reviter(r.begin()));
}

TEST(algorithms, index_into)
{
    using ivector = std::vector<std::ptrdiff_t>;
    using arb::util::size;
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

TEST(algorithms, binary_find)
{
    using arb::algorithms::binary_find;

    // empty containers
    {
        std::vector<int> v;
        EXPECT_TRUE(binary_find(v, 100) == std::end(v));
    }

    // value not present and greater than all entries
    {
        int a[] = {1, 10, 15};
        EXPECT_TRUE(binary_find(a, 100) == std::end(a));

        std::vector<int> v{1, 10, 15};
        EXPECT_TRUE(binary_find(v, 100) == std::end(v));
    }

    // value not present and less than all entries
    {
        int a[] = {1, 10, 15};
        EXPECT_TRUE(binary_find(a, -1) == std::end(a));

        std::vector<int> v{1, 10, 15};
        EXPECT_TRUE(binary_find(v, -1) == std::end(v));
    }

    // value not present and inside lower-upper bounds
    {
        int a[] = {1, 10, 15};
        EXPECT_TRUE(binary_find(a, 4) == std::end(a));

        std::vector<int> v{1, 10, 15};
        EXPECT_TRUE(binary_find(v, 4) == std::end(v));
    }

    // value is first in range
    {
        int a[] = {1, 10, 15};
        auto ita = binary_find(a, 1);
        auto found = ita!=std::end(a);
        EXPECT_TRUE(found);
        EXPECT_EQ(std::distance(std::begin(a), ita), 0);
        if (found) {
            EXPECT_EQ(*ita, 1);
        }

        std::vector<int> v{1, 10, 15};
        auto itv = binary_find(v, 1);
        found = itv!=std::end(v);
        EXPECT_TRUE(found);
        EXPECT_EQ(std::distance(std::begin(v), itv), 0);
        if (found) {
            EXPECT_EQ(*itv, 1);
        }
    }

    // value is last in range
    {
        int a[] = {1, 10, 15};
        auto ita = binary_find(a, 15);
        auto found = ita!=std::end(a);
        EXPECT_TRUE(found);
        EXPECT_EQ(std::distance(std::begin(a), ita), 2);
        if (found) {
            EXPECT_EQ(*ita, 15);
        }

        std::vector<int> v{1, 10, 15};
        auto itv = binary_find(v, 15);
        found = itv!=std::end(v);
        EXPECT_TRUE(found);
        EXPECT_EQ(std::distance(std::begin(v), itv), 2);
        if (found) {
            EXPECT_EQ(*itv, 15);
        }
    }

    // value is last present and neither first nor last in range
    {
        int a[] = {1, 10, 15};
        auto ita = binary_find(a, 10);
        auto found = ita!=std::end(a);
        EXPECT_TRUE(found);
        EXPECT_EQ(std::distance(std::begin(a), ita), 1);
        if (found) {
            EXPECT_EQ(*ita, 10);
        }

        std::vector<int> v{1, 10, 15};
        auto itv = binary_find(v, 10);
        found = itv!=std::end(v);
        EXPECT_TRUE(found);
        EXPECT_EQ(std::distance(std::begin(v), itv), 1);
        if (found) {
            EXPECT_EQ(*itv, 10);
        }
    }

    // value is last present and neither first nor last in range and range has even size
    {
        int a[] = {1, 10, 15, 27};
        auto ita = binary_find(a, 10);
        auto found = ita!=std::end(a);
        EXPECT_TRUE(found);
        EXPECT_EQ(std::distance(std::begin(a), ita), 1);
        if (found) {
            EXPECT_EQ(*ita, 10);
        }

        std::vector<int> v{1, 10, 15, 27};
        auto itv = binary_find(v, 10);
        found = itv!=std::end(v);
        EXPECT_TRUE(found);
        EXPECT_EQ(std::distance(std::begin(v), itv), 1);
        if (found) {
            EXPECT_EQ(*itv, 10);
        }
    }

    // test for const types
    // i.e. iterators returned from passing in a const reference to a container
    // can be compared to a const iterator from the container
    {
        std::vector<int> v{1, 10, 15};
        auto const& vr = v;
        auto itv = binary_find(vr, 10);
        auto found = itv!=std::end(vr);
        EXPECT_TRUE(found);
        EXPECT_EQ(std::distance(std::cbegin(v), itv), 1);
        if (found) {
            EXPECT_EQ(*itv, 10);
        }
    }
}

struct int_string {
    int value;

    friend bool operator<(const int_string& lhs, const std::string& rhs) {
        return lhs.value<std::stoi(rhs);
    }
    friend bool operator<(const std::string& lhs, const int_string& rhs) {
        return std::stoi(lhs)<rhs.value;
    }
    friend bool operator==(const int_string& lhs, const std::string& rhs) {
        return lhs.value==std::stoi(rhs);
    }
    friend bool operator==(const std::string& lhs, const int_string& rhs) {
        return std::stoi(lhs)==rhs.value;
    }
};

TEST(algorithms, binary_find_convert)
{
    using arb::algorithms::binary_find;

    std::vector<std::string> values = {"0", "10", "20", "30"};
    auto it = arb::algorithms::binary_find(values, int_string{20});

    EXPECT_TRUE(it!=values.end());
    EXPECT_TRUE(std::distance(values.begin(), it)==2u);
}
