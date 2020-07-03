#include <iterator>
#include <numeric>
#include <utility>
#include <vector>

#include <arbor/util/optional.hpp>
#include <arbor/cable_cell.hpp>
#include <arbor/cable_cell_param.hpp>
#include <arbor/morph/morphology.hpp>
#include <arbor/morph/locset.hpp>
#include <arbor/morph/mprovider.hpp>

#include "util/filter.hpp"
#include "util/rangeutil.hpp"
#include "util/span.hpp"

#include "common.hpp"
#include "unit_test_catalogue.hpp"
//#include "../common_cells.hpp"
#include "common_morphologies.hpp"

using namespace arb;
using util::make_span;

namespace {
    template <typename... A>
    locset as_locset(mlocation head, A... tail) {
        return join(locset(head), locset(tail)...);
    }
}

TEST(cv_policy, explicit_policy) {
    using namespace common_morphology;
    using L = mlocation;
    locset lset = as_locset(L{0, 0},  L{0, 0.5},  L{0, 1.},  L{1,  0.5},  L{4, 0.2});

    cv_policy pol = cv_policy_explicit(lset);
    for (auto& m: {m_reg_b6, m_mlt_b6}) {
        cable_cell cell(m);

        locset result = pol.cv_boundary_points(cell);
        EXPECT_EQ(thingify(lset, cell.provider()), thingify(result, cell.provider()));
    }
}

TEST(cv_policy, empty_morphology) {
    // Any policy applied to an empty morphology should give an empty locset,
    // with the exception of cv_policy_explicit (this is still being debated).

    using namespace common_morphology;
    using namespace cv_policy_flag;

    cv_policy policies[] = {
        cv_policy_fixed_per_branch(3),
        cv_policy_fixed_per_branch(3, interior_forks),
        cv_policy_max_extent(0.234),
        cv_policy_max_extent(0.234, interior_forks)
    };

    cable_cell cell(m_empty);
    auto empty_loclist = thingify(ls::nil(), cell.provider());

    for (auto& pol: policies) {
        EXPECT_EQ(empty_loclist, thingify(pol.cv_boundary_points(cell), cell.provider()));
    }
}

TEST(cv_policy, fixed_per_branch) {
    using namespace common_morphology;
    using namespace cv_policy_flag;
    using L = mlocation;

    // root branch only
    for (auto& morph: {m_reg_b1}) {
        cable_cell cell(morph);
        {
            // boundary fork points
            cv_policy pol = cv_policy_fixed_per_branch(4);
            locset points = pol.cv_boundary_points(cell);
            locset expected = as_locset(L{0, 0}, L{0, 0.25}, L{0, 0.5}, L{0, 0.75}, L{0, 1});
            EXPECT_EQ(thingify(expected, cell.provider()), thingify(points, cell.provider()));
        }
        {
            // interior fork points
            cv_policy pol = cv_policy_fixed_per_branch(4, interior_forks);
            locset points = pol.cv_boundary_points(cell);
            locset expected = as_locset(L{0, 0.125}, L{0, 0.375}, L{0, 0.625}, L{0, 0.875});
            EXPECT_EQ(thingify(expected, cell.provider()), thingify(points, cell.provider()));
        }
    }

    // spherical root, six branches and multiple top level branches cases:
    // expected points are the same.
    for (auto& morph: {m_mlt_b6}) {
        cable_cell cell(morph);

        {
            // boundary fork points
            cv_policy pol = cv_policy_fixed_per_branch(2);
            locset points = pol.cv_boundary_points(cell);
            locset expected = as_locset(
                L{0, 0}, L{0, 0.5}, L{0,1}, L{1, 0}, L{1, 0.5}, L{1,1}, L{2, 0}, L{2, 0.5}, L{2,1},
                L{3, 0}, L{3, 0.5}, L{3,1}, L{4, 0}, L{4, 0.5}, L{4,1}, L{5, 0}, L{5, 0.5}, L{5,1}
            );
            EXPECT_EQ(thingify(expected, cell.provider()), thingify(points, cell.provider()));
        }
        {
            // interior fork points
            cv_policy pol = cv_policy_fixed_per_branch(2, interior_forks);
            locset points = pol.cv_boundary_points(cell);
            locset expected = as_locset(
                L{0, 0.25}, L{0, 0.75}, L{1, 0.25}, L{1, 0.75}, L{2, 0.25}, L{2, 0.75},
                L{3, 0.25}, L{3, 0.75}, L{4, 0.25}, L{4, 0.75}, L{5, 0.25}, L{5, 0.75}
            );
            EXPECT_EQ(thingify(expected, cell.provider()), thingify(points, cell.provider()));
        }
    }
}

TEST(cv_policy, max_extent) {
    using namespace common_morphology;
    using namespace cv_policy_flag;
    using L = mlocation;

    // root branch only
    for (auto& morph: {m_reg_b1}) {
        cable_cell cell(morph);
        ASSERT_EQ(1.0, cell.embedding().branch_length(0));

        {
            // extent of 0.25 should give exact fp calculation, giving
            // 4 CVs on the root branch.
            cv_policy pol = cv_policy_max_extent(0.25);
            locset points = pol.cv_boundary_points(cell);
            locset expected = as_locset(L{0, 0}, L{0, 0.25}, L{0, 0.5}, L{0, 0.75}, L{0, 1});
            EXPECT_EQ(thingify(expected, cell.provider()), thingify(points, cell.provider()));
        }
        {
            cv_policy pol = cv_policy_max_extent(0.25, interior_forks);
            locset points = pol.cv_boundary_points(cell);
            locset expected = as_locset(L{0, 0.125}, L{0, 0.375}, L{0, 0.625}, L{0, 0.875});
            EXPECT_EQ(thingify(expected, cell.provider()), thingify(points, cell.provider()));
        }
    }

    // cell with varying branch lengths; extent not exact fraction.
    {
        cable_cell cell(m_mlt_b6);
        ASSERT_EQ(1.0, cell.embedding().branch_length(0));
        ASSERT_EQ(1.0, cell.embedding().branch_length(1));
        ASSERT_EQ(2.0, cell.embedding().branch_length(2));
        ASSERT_EQ(4.0, cell.embedding().branch_length(3));
        ASSERT_EQ(1.0, cell.embedding().branch_length(4));
        ASSERT_EQ(2.0, cell.embedding().branch_length(5));

        {
            // max extent of 0.6 should give two CVs on branches of length 1,
            // four CVs on branches of length 2, and seven CVs on the branch of length 4.
            cv_policy pol = cv_policy_max_extent(0.6);
            mlocation_list points = thingify(pol.cv_boundary_points(cell), cell.provider());

            mlocation_list points_b012 = util::assign_from(util::filter(points, [](mlocation l) { return l.branch<3; }));
            mlocation_list expected_b012 = {
                {0, 0},  {0, 0.5},  {0, 1},
                {1, 0},  {1, 0.5},  {1, 1},
                {2, 0},  {2, 0.25}, {2, 0.5}, {2, 0.75}, {2, 1}
            };
            EXPECT_EQ(expected_b012, points_b012);

            mlocation_list points_b3 = util::assign_from(util::filter(points, [](mlocation l) { return l.branch==3; }));
            EXPECT_EQ(8u, points_b3.size());
        }
    }
}

TEST(cv_policy, every_segment) {
    using namespace cv_policy_flag;

    // Cell with root branch and two child branches, with multiple samples per branch.
    // Fork is at (0., 0., 4.0).

    std::vector<mpoint> mp;
    mp.push_back({  0.,   0., 0., 0.5});

    arb::segment_tree tree;
    tree.append(mnpos, {0, 0, 0, 0.5}, {0, 0, 1, 0.5}, 1);
    tree.append(    0, {0, 0, 2, 0.5}, 1);
    tree.append(    1, {0, 0, 3, 0.5}, 1);
    tree.append(    2, {0, 0, 4, 0.5}, 1);

    tree.append(    3, {0, 1, 4, 0.5}, 1);
    tree.append(    4, {0, 2, 4, 0.5}, 1);
    tree.append(    5, {0, 3, 4, 0.5}, 1);
    tree.append(    6, {0, 4, 4, 0.5}, 1);

    tree.append(    3, {1, 0, 4, 0.5}, 1);
    tree.append(    8, {2, 0, 4, 0.5}, 1);
    tree.append(    9, {3, 0, 4, 0.5}, 1);
    tree.append(   10, {4, 0, 4, 0.5}, 1);

    morphology m(tree);

    // Including all samples:
    {
        cable_cell cell(m);
        cv_policy pol = cv_policy_every_segment();
        mlocation_list points = thingify(pol.cv_boundary_points(cell), cell.provider());
        util::sort(points);

        mlocation_list expected = {
            {0, 0}, {0, 0.25}, {0, 0.5}, {0, 0.75}, {0, 1.},
            {1, 0}, {1, 0.25}, {1, 0.5}, {1, 0.75}, {1, 1.},
            {2, 0}, {2, 0.25}, {2, 0.5}, {2, 0.75}, {2, 1.}
        };

        EXPECT_EQ(expected, points);
    }
}
