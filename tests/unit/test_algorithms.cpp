#include <iterator>
#include <random>
#include <vector>

#include "../gtest.h"

#include <algorithms.hpp>
#include "../test_util.hpp"
#include <util/debug.hpp>
#include <util/meta.hpp>

/// tests the sort implementation in threading
/// is only parallel if TBB is being used
TEST(algorithms, parallel_sort)
{
    auto n = 10000;
    std::vector<int> v(n);
    std::iota(v.begin(), v.end(), 1);

    // intialize with the default random seed
    std::shuffle(v.begin(), v.end(), std::mt19937());

    // assert that the original vector has in fact been permuted
    EXPECT_FALSE(std::is_sorted(v.begin(), v.end()));

    arb::threading::sort(v);

    EXPECT_TRUE(std::is_sorted(v.begin(), v.end()));
    for(auto i=0; i<n; ++i) {
       EXPECT_EQ(i+1, v[i]);
   }
}


TEST(algorithms, sum)
{
    // sum of 10 times 2 is 20
    std::vector<int> v1(10, 2);
    EXPECT_EQ(10*2, arb::algorithms::sum(v1));

    // make an array 1:20 and sum it up using formula for arithmetic sequence
    auto n = 20;
    std::vector<int> v2(n);
    // can't use iota because the Intel compiler optimizes it out, despite
    // the result being required in EXPECT_EQ
    // std::iota(v2.begin(), v2.end(), 1);
    for(auto i=0; i<n; ++i) { v2[i] = i+1; }
    EXPECT_EQ((n+1)*n/2, arb::algorithms::sum(v2));
}

TEST(algorithms, make_index)
{
    {
        std::vector<int> v(10, 1);
        auto index = arb::algorithms::make_index(v);

        EXPECT_EQ(index.size(), 11u);
        EXPECT_EQ(index.front(), 0);
        EXPECT_EQ(index.back(), arb::algorithms::sum(v));
    }

    {
        std::vector<int> v(10);
        std::iota(v.begin(), v.end(), 1);
        auto index = arb::algorithms::make_index(v);

        EXPECT_EQ(index.size(), 11u);
        EXPECT_EQ(index.front(), 0);
        EXPECT_EQ(index.back(), arb::algorithms::sum(v));
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

TEST(algorithms, is_positive)
{
    EXPECT_TRUE(
        arb::algorithms::is_positive(
            std::vector<int>{}
        )
    );
    EXPECT_TRUE(
        arb::algorithms::is_positive(
            std::vector<int>{3, 2, 1}
        )
    );
    EXPECT_FALSE(
        arb::algorithms::is_positive(
            std::vector<int>{3, 2, 1, 0}
        )
    );
    EXPECT_FALSE(
        arb::algorithms::is_positive(
            std::vector<int>{-1}
        )
    );
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

TEST(algorithms, is_sorted)
{
    EXPECT_TRUE(
        arb::algorithms::is_sorted(
            std::vector<int>{}
        )
    );
    EXPECT_TRUE(
        arb::algorithms::is_sorted(
            std::vector<int>{100}
        )
    );
    EXPECT_TRUE(
        arb::algorithms::is_sorted(
            std::vector<int>{0,1,2}
        )
    );
    EXPECT_TRUE(
        arb::algorithms::is_sorted(
            std::vector<int>{0,2,100}
        )
    );
    EXPECT_TRUE(
        arb::algorithms::is_sorted(
            std::vector<int>{0,0}
        )
    );
    EXPECT_TRUE(
        arb::algorithms::is_sorted(
            std::vector<int>{0,1,2,2,2,2,3,4,5,5,5}
        )
    );
    EXPECT_FALSE(
        arb::algorithms::is_sorted(
            std::vector<int>{0,1,2,1}
        )
    );
    EXPECT_FALSE(
        arb::algorithms::is_sorted(
            std::vector<int>{1,0}
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

TEST(algorithms, index_into)
{
    using C = std::vector<int>;
    using arb::util::size;

    // by default index_into assumes that the inputs satisfy
    // quite a strong set of prerequisites
    auto tests = {
        std::make_pair(C{}, C{}),
        std::make_pair(C{100}, C{}),
        std::make_pair(C{0,1,3,4,6,7,10,11}, C{0,4,6,7,11}),
        std::make_pair(C{0,1,3,4,6,7,10,11}, C{0}),
        std::make_pair(C{0,1,3,4,6,7,10,11}, C{11}),
        std::make_pair(C{0,1,3,4,6,7,10,11}, C{4}),
        std::make_pair(C{0,1,3,4,6,7,10,11}, C{0,11}),
        std::make_pair(C{0,1,3,4,6,7,10,11}, C{4,11}),
        std::make_pair(C{0,1,3,4,6,7,10,11}, C{0,1,3,4,6,7,10,11})
    };

    test_index_into tester;
    for(auto& t : tests) {
        EXPECT_TRUE(
            tester(t.second, t.first, arb::algorithms::index_into(t.second, t.first))
        );
    }

    // test for arrays
    int sub[] = {2, 3, 5, 9};
    int sup[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto idx = arb::algorithms::index_into(sub, sup);
    EXPECT_EQ(size(sub), size(idx));
    auto it = idx.begin();
    for (auto i: sub) {
        EXPECT_EQ(i, *it++);
    }
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
        EXPECT_EQ(std::distance(std::begin(a), ita), 0u);
        if (found) {
            EXPECT_EQ(*ita, 1);
        }

        std::vector<int> v{1, 10, 15};
        auto itv = binary_find(v, 1);
        found = itv!=std::end(v);
        EXPECT_TRUE(found);
        EXPECT_EQ(std::distance(std::begin(v), itv), 0u);
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
        EXPECT_EQ(std::distance(std::begin(a), ita), 2u);
        if (found) {
            EXPECT_EQ(*ita, 15);
        }

        std::vector<int> v{1, 10, 15};
        auto itv = binary_find(v, 15);
        found = itv!=std::end(v);
        EXPECT_TRUE(found);
        EXPECT_EQ(std::distance(std::begin(v), itv), 2u);
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
        EXPECT_EQ(std::distance(std::begin(a), ita), 1u);
        if (found) {
            EXPECT_EQ(*ita, 10);
        }

        std::vector<int> v{1, 10, 15};
        auto itv = binary_find(v, 10);
        found = itv!=std::end(v);
        EXPECT_TRUE(found);
        EXPECT_EQ(std::distance(std::begin(v), itv), 1u);
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
        EXPECT_EQ(std::distance(std::begin(a), ita), 1u);
        if (found) {
            EXPECT_EQ(*ita, 10);
        }

        std::vector<int> v{1, 10, 15, 27};
        auto itv = binary_find(v, 10);
        found = itv!=std::end(v);
        EXPECT_TRUE(found);
        EXPECT_EQ(std::distance(std::begin(v), itv), 1u);
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
        EXPECT_EQ(std::distance(arb::util::cbegin(v), itv), 1u);
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
