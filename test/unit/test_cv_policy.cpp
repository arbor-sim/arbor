#include <iterator>
#include <numeric>
#include <utility>
#include <vector>

#include <arbor/cable_cell.hpp>
#include <arbor/cable_cell_param.hpp>
#include <arbor/morph/morphology.hpp>
#include <arbor/morph/locset.hpp>
#include <arbor/morph/mprovider.hpp>
#include <arbor/morph/region.hpp>

#include "util/filter.hpp"
#include "util/rangeutil.hpp"
#include "util/span.hpp"

#include "../common_cells.hpp"
#include "common_morphologies.hpp"
#include "morph_pred.hpp"

using namespace arb;
using util::make_span;
using testing::locset_eq;
using testing::region_eq;
using testing::mlocationlist_eq;

using namespace common_morphology;

namespace {
    template <typename... A>
    locset as_locset(mlocation head, A... tail) {
        return join(locset(head), locset(tail)...);
    }
}

TEST(cv_policy, single) {
    // In all instances, expect the boundary points to correspond to
    // the extremal points of the completions of the components of the
    // supplied region.


    cable_cell cell(m_mlt_b6, {});
    for (region reg:
            {reg::all(), reg::branch(2), reg::cable(3, 0.25, 1.),
             join(reg::cable(1, 0.75, 1), reg::branch(3), reg::cable(2, 0, 0.5)),
             join(reg::cable(2, 0, 0.5), reg::branch(3), reg::cable(4, 0, 0.5))})
    {
        locset expected = ls::cboundary(reg);
        EXPECT_TRUE(locset_eq(cell.provider(), ls::cboundary(reg), cv_policy_single(reg).cv_boundary_points(cell)));
    }

    EXPECT_TRUE(locset_eq(cell.provider(), cv_policy_single().cv_boundary_points(cell), cv_policy_single(reg::all()).cv_boundary_points(cell)));
}

TEST(cv_policy, explicit_policy) {
    using L = mlocation;
    locset lset = as_locset(L{0, 0},  L{0, 0.5},  L{0, 1.},  L{1,  0.5},  L{4, 0.2});

    cv_policy pol = cv_policy_explicit(lset);
    for (auto& m: {m_reg_b6, m_mlt_b6}) {
        cable_cell cell(m, {});

        locset result = pol.cv_boundary_points(cell);
        locset expected = join(ls::boundary(reg::all()), lset);
        EXPECT_TRUE(locset_eq(cell.provider(), expected, result));
    }

    // With cables 1 and 2, expect to pick up (1, 0.5) from locset,
    // and cable ends (1, 0), (1, 1), (2, 0), (2, 1), as the two
    // cables constitute two components.

    region b12 = join(reg::branch(1), reg::branch(2));
    pol = cv_policy_explicit(lset, b12);
    for (auto& m: {m_reg_b6, m_mlt_b6}) {
        cable_cell cell(m, {});

        locset result = pol.cv_boundary_points(cell);
        locset expected = as_locset(L{1, 0}, L{1, 0.5}, L{1, 1}, L{2, 0}, L{2, 1});
        EXPECT_TRUE(locset_eq(cell.provider(), expected, result));
    }

    // Taking the completion of the two cables, the boundary of the region
    // will be (0, 1), (1, 1), (2, 1) for m_mlt_b6.

    pol = cv_policy_explicit(lset, reg::complete(b12));
    for (auto& m: {m_mlt_b6}) {
        cable_cell cell(m, {});

        locset result = pol.cv_boundary_points(cell);
        locset expected = as_locset(L{0, 1}, L{1, 0.5}, L{1, 1}, L{2, 1});
        EXPECT_TRUE(locset_eq(cell.provider(), expected, result));
    }
}

TEST(cv_policy, empty_morphology) {
    // Any policy applied to an empty morphology should give an empty locset.

    using namespace cv_policy_flag;

    cv_policy policies[] = {
        cv_policy_fixed_per_branch(3),
        cv_policy_fixed_per_branch(3, interior_forks),
        cv_policy_max_extent(0.234),
        cv_policy_max_extent(0.234, interior_forks),
        cv_policy_single(),
        cv_policy_single(reg::all()),
        cv_policy_explicit(ls::location(0, 0))
    };


    cable_cell cell(m_empty, {});

    for (auto& pol: policies) {
        EXPECT_TRUE(locset_eq(cell.provider(), ls::nil(), pol.cv_boundary_points(cell)));
    }
}

TEST(cv_policy, fixed_per_branch) {
    using namespace cv_policy_flag;
    using L = mlocation;

    // Root branch only:
    {
        cable_cell cell(m_reg_b1, {});
        {
            // boundary fork points
            cv_policy pol = cv_policy_fixed_per_branch(4);
            locset expected = as_locset(L{0, 0}, L{0, 0.25}, L{0, 0.5}, L{0, 0.75}, L{0, 1});
            EXPECT_TRUE(locset_eq(cell.provider(), expected, pol.cv_boundary_points(cell)));
        }
        {
            // interior fork points
            cv_policy pol = cv_policy_fixed_per_branch(4, interior_forks);
            locset points = pol.cv_boundary_points(cell);
            locset expected = as_locset(L{0, 0}, L{0, 0.125}, L{0, 0.375}, L{0, 0.625}, L{0, 0.875}, L{0, 1});
            EXPECT_TRUE(locset_eq(cell.provider(), expected, pol.cv_boundary_points(cell)));
        }
    }

    // Multiple top level branches:
    // top level branches are 0 and 3, terminal branches are 1, 2, 4 and 5.
    {
        cable_cell cell(m_mlt_b6, {});
        {
            // With boundary fork points:
            cv_policy pol = cv_policy_fixed_per_branch(2);
            locset expected = as_locset(
                L{0, 0}, L{0, 0.5}, L{0,1}, L{1, 0}, L{1, 0.5}, L{1,1}, L{2, 0}, L{2, 0.5}, L{2,1},
                L{3, 0}, L{3, 0.5}, L{3,1}, L{4, 0}, L{4, 0.5}, L{4,1}, L{5, 0}, L{5, 0.5}, L{5,1}
            );
            EXPECT_TRUE(locset_eq(cell.provider(), expected, pol.cv_boundary_points(cell)));
        }
        {
            // With interior fork points:
            cv_policy pol = cv_policy_fixed_per_branch(2, interior_forks);
            locset expected = as_locset(
                L{0, 0}, L{0, 0.25}, L{0, 0.75},
                L{1, 0.25}, L{1, 0.75}, L{1, 1.0},
                L{2, 0.25}, L{2, 0.75}, L{2, 1.0},
                L{3, 0}, L{3, 0.25}, L{3, 0.75},
                L{4, 0.25}, L{4, 0.75}, L{4, 1.0},
                L{5, 0.25}, L{5, 0.75}, L{5, 1.0}
            );
            EXPECT_TRUE(locset_eq(cell.provider(), expected, pol.cv_boundary_points(cell)));
        }
    }

    // Restrict to an incomplete subtree (distal half of branch 0 and all of branch 2)
    // in m_mlt_b6 morphology.
    {
        cable_cell cell(m_mlt_b6, {});
        region reg = mcable_list{{0, 0.5, 1.}, {2, 0., 1.}};
        {
            // With two per branch and fork points as boundaries, expect to see:
            //     (0, 0.5), (0, 0.75), (0, 1) on branch 0;
            //     (2, 0), (2, 0.5), (2, 1) on branch 2;
            //     (1, 0) on branch 1.
            cv_policy pol = cv_policy_fixed_per_branch(2, reg);
            locset expected = as_locset(
                L{0, 0.5}, L{0, 0.75}, L{0, 1},
                L{1, 0},
                L{2, 0}, L{2, 0.5}, L{2, 1}
            );
            EXPECT_TRUE(locset_eq(cell.provider(), expected, pol.cv_boundary_points(cell)));
        }
        {
            // With two per branch and interior forks, expect to see:
            //     (0, 0.5), (0, 0.625), (0, 0.0875) on branch 0;
            //     (2, 0.25), (2, 0.75), (2, 1) on branch 2;
            //     (1, 0) on branch 1.
            cv_policy pol = cv_policy_fixed_per_branch(2, reg, interior_forks);
            locset expected = as_locset(
                L{0, 0.5}, L{0, 0.625}, L{0, 0.875},
                L{1, 0},
                L{2, 0.25}, L{2, 0.75}, L{2, 1}
            );
            EXPECT_TRUE(locset_eq(cell.provider(), expected, pol.cv_boundary_points(cell)));
        }
    }
}

TEST(cv_policy, max_extent) {
    using namespace cv_policy_flag;
    using L = mlocation;

    // Root branch only:
    {
        cable_cell cell(m_reg_b1, {});
        ASSERT_EQ(1.0, cell.embedding().branch_length(0));

        {
            // Extent of 0.25 should give exact fp calculation, giving
            // 4 CVs on the root branch.
            cv_policy pol = cv_policy_max_extent(0.25);
            locset expected = as_locset(L{0, 0}, L{0, 0.25}, L{0, 0.5}, L{0, 0.75}, L{0, 1});
            EXPECT_TRUE(locset_eq(cell.provider(), expected, pol.cv_boundary_points(cell)));
        }
        {
            // Same, but applied to cable (0, 0.25, 0.75) should give 2 Cvs.
            cv_policy pol = cv_policy_max_extent(0.25, reg::cable(0, 0.25, 0.75));
            locset expected = as_locset(L{0, 0.25}, L{0, 0.5}, L{0, 0.75});
            EXPECT_TRUE(locset_eq(cell.provider(), expected, pol.cv_boundary_points(cell)));

        }
        {
            // Interior forks:
            cv_policy pol = cv_policy_max_extent(0.25, interior_forks);
            locset expected = as_locset(L{0, 0}, L{0, 0.125}, L{0, 0.375}, L{0, 0.625}, L{0, 0.875}, L{0, 1});
            EXPECT_TRUE(locset_eq(cell.provider(), expected, pol.cv_boundary_points(cell)));
        }
        {
            // Interior forks but restricted to sub-cable.
            cv_policy pol = cv_policy_max_extent(0.25, reg::cable(0, 0.25, 0.75), interior_forks);
            locset expected = as_locset(L{0, 0.25}, L{0, 0.375}, L{0, 0.625}, L{0, 0.75});
            EXPECT_TRUE(locset_eq(cell.provider(), expected, pol.cv_boundary_points(cell)));

        }
    }

    // Cell with varying branch lengths; extent not exact fraction:
    {
        cable_cell cell(m_mlt_b6, {});
        ASSERT_EQ(1.0, cell.embedding().branch_length(0));
        ASSERT_EQ(1.0, cell.embedding().branch_length(1));
        ASSERT_EQ(2.0, cell.embedding().branch_length(2));
        ASSERT_EQ(4.0, cell.embedding().branch_length(3));
        ASSERT_EQ(1.0, cell.embedding().branch_length(4));
        ASSERT_EQ(2.0, cell.embedding().branch_length(5));

        {
            // Max extent of 0.6 should give two CVs on branches of length 1,
            // four CVs on branches of length 2, and seven CVs on the branch of length 4.
            cv_policy pol = cv_policy_max_extent(0.6);
            mlocation_list points = thingify(pol.cv_boundary_points(cell), cell.provider());

            mlocation_list points_b012 = util::assign_from(util::filter(points, [](mlocation l) { return l.branch<3; }));
            mlocation_list expected_b012 = {
                {0, 0},  {0, 0.5},  {0, 1},
                {1, 0},  {1, 0.5},  {1, 1},
                {2, 0},  {2, 0.25}, {2, 0.5}, {2, 0.75}, {2, 1}
            };
            EXPECT_TRUE(mlocationlist_eq(expected_b012, points_b012));

            mlocation_list points_b3 = util::assign_from(util::filter(points, [](mlocation l) { return l.branch==3; }));
            EXPECT_EQ(8u, points_b3.size());
        }
    }
}

TEST(cv_policy, every_segment) {
    using namespace cv_policy_flag;

    // Cell with root branch and two child branches, with multiple samples per branch.
    // Fork is at (0., 0., 4.0).

    std::vector<mpoint> points;

    points.push_back({  0.,   0., 0., 0.5});
    for (auto i: make_span(4)) points.push_back({  0.,   0., i+1., 0.5});
    for (auto i: make_span(4)) points.push_back({  0., i+1.,  4.0, 0.5});
    for (auto i: make_span(4)) points.push_back({i+1.,    0,  4.0, 0.5});

    std::vector<msize_t> parents = {mnpos, 0, 1, 2, 3, 4, 5, 6, 7, 4, 9, 10, 11 };
    auto tree = segments_from_points(points, parents);
    morphology m{tree};

    // Including all samples:
    {
        cable_cell cell(m, {});
        cv_policy pol = cv_policy_every_segment();

        mlocation_list expected = {
            {0, 0}, {0, 0.25}, {0, 0.5}, {0, 0.75}, {0, 1},
            {1, 0}, {1, 0.25}, {1, 0.5}, {1, 0.75}, {1, 1},
            {2, 0}, {2, 0.25}, {2, 0.5}, {2, 0.75}, {2, 1}
        };

        EXPECT_TRUE(locset_eq(cell.provider(), locset(expected), pol.cv_boundary_points(cell)));
    }
    // Restricting to the two child branches (disconnected):
    {
        cable_cell cell(m, {});
        region reg = join(reg::branch(1), reg::branch(2));
        cv_policy pol = cv_policy_every_segment(reg);

        // Get samples from branches 1 and 2, plus boundary points from completions of each
        // branch, viz. (0, 1), (2, 0), (1, 1) from branch 1 and (0, 1), (1, 0), (2, 1) from
        // branch 2.
        mlocation_list expected = {
            {0, 1},
            {1, 0}, {1, 0.25}, {1, 0.5}, {1, 0.75}, {1, 1},
            {2, 0}, {2, 0.25}, {2, 0.5}, {2, 0.75}, {2, 1}
        };

        EXPECT_TRUE(locset_eq(cell.provider(), locset(expected), pol.cv_boundary_points(cell)));
    }
}

TEST(cv_policy, domain) {
    using namespace cv_policy_flag;

    region reg1 = join(reg::branch(1), reg::cable(2, 0, 0.5));
    region reg2 = join(reg::branch(1), reg::cable(2, 0.5, 1), reg::cable(4, 0, 1));

    cable_cell cell(m_mlt_b6, {});

    EXPECT_TRUE(region_eq(cell.provider(), reg1, cv_policy_single(reg1).domain()));
    EXPECT_TRUE(region_eq(cell.provider(), reg1, cv_policy_fixed_per_branch(3, reg1).domain()));
    EXPECT_TRUE(region_eq(cell.provider(), reg1, cv_policy_fixed_per_branch(3, reg1, interior_forks).domain()));
    EXPECT_TRUE(region_eq(cell.provider(), reg1, cv_policy_max_extent(3, reg1).domain()));
    EXPECT_TRUE(region_eq(cell.provider(), reg1, cv_policy_max_extent(3, reg1, interior_forks).domain()));
    EXPECT_TRUE(region_eq(cell.provider(), reg1, cv_policy_every_segment(reg1).domain()));

    EXPECT_TRUE(region_eq(cell.provider(), join(reg1, reg2), (cv_policy_single(reg1)+cv_policy_single(reg2)).domain()));
    EXPECT_TRUE(region_eq(cell.provider(), join(reg1, reg2), (cv_policy_single(reg1)|cv_policy_single(reg2)).domain()));
}

TEST(cv_policy, combinators) {
    auto unique_sum = [](auto... a) { return ls::support(sum(locset(a)...)); };

    cable_cell cell(m_reg_b6, {});
    auto eval_locset_eq = [&cell](const locset& a, const cv_policy& p) {
        return locset_eq(cell.provider(), a, p.cv_boundary_points(cell));
    };

    {
        mlocation_list locs1{{0, 0.5}, {1, 0.25}, {2, 1}};
        mlocation_list locs2{{0, 0.75}, {1, 0.25}, {4, 0}};
        locset all_bdy = ls::boundary(reg::all());

        ASSERT_TRUE(eval_locset_eq(unique_sum(all_bdy, locs1), cv_policy_explicit(locs1)));
        ASSERT_TRUE(eval_locset_eq(unique_sum(all_bdy, locs2), cv_policy_explicit(locs2)));

        EXPECT_TRUE(eval_locset_eq(unique_sum(all_bdy, locs1, locs2), cv_policy_explicit(locs1)+cv_policy_explicit(locs2)));
        EXPECT_TRUE(eval_locset_eq(unique_sum(all_bdy, locs2), cv_policy_explicit(locs1)|cv_policy_explicit(locs2)));
    }

    {
        region reg12 = join(reg::branch(1), reg::branch(2));
        region reg23 = join(reg::branch(2), reg::branch(3));

        cv_policy pol12 = cv_policy_explicit(ls::on_branches(0.5), reg12);
        cv_policy pol23 = cv_policy_explicit(ls::on_branches(0.75), reg23);

        using L = mlocation;

        ASSERT_TRUE(eval_locset_eq(unique_sum(ls::boundary(reg12), L{1, 0.5}, L{2, 0.5}), pol12));
        ASSERT_TRUE(eval_locset_eq(unique_sum(ls::boundary(reg23), L{2, 0.75}, L{3, 0.75}), pol23));

        EXPECT_TRUE(eval_locset_eq(unique_sum(ls::boundary(reg12), ls::boundary(reg23), L{1, 0.5}, L{2, 0.5}, L{2, 0.75}, L{3, 0.75}),
            pol12+pol23));

        EXPECT_TRUE(eval_locset_eq(unique_sum(ls::boundary(reg12), ls::boundary(reg23), L{1, 0.5}, L{2, 0.75}, L{3, 0.75}),
            pol12|pol23));
    }
}
