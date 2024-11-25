#include <algorithm>
#include <utility>

#include <arbor/cable_cell.hpp>
#include <arbor/morph/cv_data.hpp>
#include <arbor/morph/morphology.hpp>
#include <arbor/morph/locset.hpp>
#include <arbor/morph/region.hpp>

#include <arborio/label_parse.hpp>

#include "fvm_layout.hpp"
#include "util/rangeutil.hpp"

#include "common.hpp"
#include "common_morphologies.hpp"
#include "morph_pred.hpp"
#include "../common_cells.hpp"

using namespace arb;
using util::make_span;

::testing::AssertionResult verify_cv_children(const cv_geometry& g) {
    unsigned visited_children = 0;
    for (unsigned i = 0; i<g.size(); ++i) {
        if (!util::is_sorted(g.children(i))) {
            return ::testing::AssertionFailure() << "CV " << i
                << " has unsorted sequence of child CVs";
        }

        for (auto cv: g.children(i)) {
            if ((arb_index_type)i != g.cv_parent.at(cv)) {
                return ::testing::AssertionFailure() << "CV " << i
                    << " has child CV " << cv << " which has parent " << g.cv_parent.at(cv);
            }
            ++visited_children;
        }
    }

    if (g.cv_children.size()!=visited_children) {
        return ::testing::AssertionFailure() << "geometry child CV count " << g.cv_children.size()
            << " does not equal number of visited children " << visited_children;
    }

    unsigned n_nonempty_cells = 0;
    for (auto c: util::make_span(g.n_cell())) {
        if (!g.cell_cvs(c).empty()) {
            ++n_nonempty_cells;
        }
    }
    if (g.cv_children.size()!=g.size()-n_nonempty_cells) {
        return ::testing::AssertionFailure() << "child CV count " << g.cv_children.size()
            << " plus root CV count " << n_nonempty_cells
            << " does not equal total number of CVs " << g.size();
    }
    return ::testing::AssertionSuccess();
}

namespace arb {
namespace cv_prefer {
std::ostream& operator<<(std::ostream& out, ::arb::cv_prefer::type p) {
    switch (p) {
    case cv_proximal: return out << "cv_proximal";
    case cv_distal: return out << "cv_distal";
    case cv_empty: return out << "cv_empty";
    case cv_nonempty: return out << "cv_nonempty";
    default: return out;
    }
}
}
}

TEST(cv_geom, empty) {
    using namespace common_morphology;

    cable_cell empty_cell{m_empty, {}};
    cv_geometry geom(empty_cell, ls::nil());
    EXPECT_TRUE(verify_cv_children(geom));

    EXPECT_TRUE(geom.cv_parent.empty());
    EXPECT_TRUE(geom.cv_cables.empty());
    EXPECT_TRUE(geom.cv_cables_divs.empty());

    EXPECT_EQ(0u, geom.size()); // size()/empty() reflects number of CVs.
    EXPECT_EQ(1u, geom.n_cell()); // can have no CVs but >0 cells.
}

static bool region_eq(const mprovider& p, region a, region b) {
    return thingify(a, p)==thingify(b, p);
}

TEST(cv_geom, trivial) {
    using namespace common_morphology;

    for (auto& p: test_morphologies) {
        if (p.second.empty()) continue;

        SCOPED_TRACE(p.first);
        cable_cell cell{p.second, {}};
        auto& m = cell.morphology();

        // Equivalent ways of specifying one CV comprising whole cell:
        cv_geometry geom1(cell, ls::nil());
        cv_geometry geom2(cell, ls::terminal());

        EXPECT_TRUE(verify_cv_children(geom1));
        EXPECT_TRUE(verify_cv_children(geom2));

        EXPECT_EQ(1u, geom1.size());
        EXPECT_EQ(geom1.cv_cables, geom2.cv_cables);

        // These are equivalent too, if there is a single root branch.
        cv_geometry geom3(cell, ls::root());
        cv_geometry geom4(cell, join(ls::root(), ls::terminal()));

        EXPECT_TRUE(verify_cv_children(geom3));
        EXPECT_TRUE(verify_cv_children(geom4));

        EXPECT_EQ(geom3.cv_cables, geom4.cv_cables);
        if (m.branch_children(mnpos).size()==1) {
            EXPECT_EQ(geom1.cv_cables, geom4.cv_cables);
        }

        mcable_list geom1_cables = util::assign_from(geom1.cables(0));
        EXPECT_TRUE(region_eq(cell.provider(), reg::all(), geom1_cables));
    }
}

TEST(cv_geom, one_cv_per_branch) {
    using namespace common_morphology;

    auto complete = [] (const arb::morphology& m, arb::mcable c) {
        return thingify(arb::reg::complete(arb::region(c)), arb::mprovider(m)).cables();
    };

    for (auto& p: test_morphologies) {
        if (p.second.empty()) continue;
        SCOPED_TRACE(p.first);

        cable_cell cell{p.second, {}};
        auto& m = cell.morphology();

        auto cell_cv_geom = cv_geometry(cell, sum(ls::on_branches(0), ls::on_branches(1)));
        auto geom = cv_geometry(cell_cv_geom);
        EXPECT_TRUE(verify_cv_children(geom));

        // Expect trivial CVs at every fork point, and single-cable CVs for each branch.
        std::vector<unsigned> seen_branches(m.num_branches(), 0);
        auto n_branch_child = [&m](msize_t b) { return m.branch_children(b).size(); };
        for (auto i: make_span(geom.size())) {
            auto cables = geom.cables(i);

            auto c = cables.front();

            if (c.prox_pos==c.dist_pos) {
                EXPECT_LT(1u, cables.size());
                if (c.branch==0 && c.prox_pos==0) {
                    EXPECT_TRUE(n_branch_child(mnpos)>1);
                }
                else {
                    EXPECT_EQ(1., c.prox_pos);
                    EXPECT_TRUE(n_branch_child(c.branch)>1);
                }
                // Cables in trivial CV should be the same as those in the extent over the point.
                EXPECT_TRUE(testing::seq_eq(complete(m,c), cables));
            }
            else {
                ASSERT_EQ(1u, cables.size());
                ++seen_branches[c.branch];
                EXPECT_EQ(1., seen_branches[c.branch]);
                EXPECT_EQ(0., c.prox_pos);
                EXPECT_EQ(1., c.dist_pos);

                // Confirm parent CV is fork CV:
                if (i>0) {
                    auto fork_ext = complete(m, {c.branch, 0});
                    mcable_list pcables = util::assign_from(geom.cables(geom.cv_parent[i]));
                    ASSERT_TRUE(testing::cablelist_eq(fork_ext, pcables));
                }
            }
        }

        EXPECT_TRUE(std::find(seen_branches.begin(), seen_branches.end(), 0)==seen_branches.end());
    }
}

TEST(cv_geom, midpoints) {
    using namespace common_morphology;

    // Place CV boundaries at the midpoints of each branch.
    for (auto& p: test_morphologies) {
        if (p.second.empty()) continue;
        SCOPED_TRACE(p.first);

        cable_cell cell{p.second, {}};
        auto& m = cell.morphology();

        cv_geometry geom(cell, ls::on_branches(0.5));
        EXPECT_TRUE(verify_cv_children(geom));

        // Expect CVs to be either: covering fork points, with one cable per branch
        // at the fork (for a multiple-root-branch morphology, this would be treating
        // (0, 0) as a fork); or the last halves of terminal branches or the first half
        // of a unique root branch.

        auto n_branch_child = [&m](msize_t b) { return m.branch_children(b).size(); };
        for (auto i: make_span(geom.size())) {
            auto cables = geom.cables(i);

            if (i==0) {
                // Expect inital half of single branch cell, or branched CV around (0,0).
                if (cables.size()==1) {
                    EXPECT_EQ(1u, n_branch_child(mnpos));
                    auto c = cables.front();
                    EXPECT_EQ(0u, c.branch);
                    EXPECT_EQ(0.0, c.prox_pos);
                    EXPECT_EQ(0.5, c.dist_pos);
                }
                else {
                    EXPECT_TRUE(n_branch_child(mnpos)>1);
                    for (auto& c: cables) {
                        auto x = canonical(m, mlocation{c.branch, 0.});
                        EXPECT_EQ(0u, x.branch);

                        EXPECT_EQ(0.0, c.prox_pos);
                        EXPECT_EQ(0.5, c.dist_pos);
                    }
                }
            }
            else {
                // Expect final half of terminal branch or a branched CV around an interior fork.
                if (cables.size()==1) {
                    // Terminal segment, or initial segment of 1-branch cell.
                    auto c = cables.front();
                    EXPECT_EQ(0.5, c.prox_pos);
                    EXPECT_EQ(1.0, c.dist_pos);
                    EXPECT_EQ(0u, n_branch_child(c.branch));
                }
                else {
                    auto prox_cable = cables.front();
                    EXPECT_EQ(0.5, prox_cable.prox_pos);
                    EXPECT_EQ(1.0, prox_cable.dist_pos);

                    msize_t prox_branch = prox_cable.branch;
                    EXPECT_EQ(1+n_branch_child(prox_branch), cables.size());

                    for (unsigned j = 1; j<cables.size(); ++j) {
                        auto& c = cables[j];
                        EXPECT_EQ(0.0, c.prox_pos);
                        EXPECT_EQ(0.5, c.dist_pos);
                        auto x = canonical(m, mlocation{c.branch, 0.});
                        EXPECT_EQ(prox_branch, x.branch);
                    }
                }
            }
        }
    }
}

TEST(cv_geom, weird) {
    // m_reg_b6 has the following branch structure:
    //
    // ---0---+---1---+---3---
    //        |       |
    //        |       +---4---
    //        2       |
    //        |       +---5---
    //        |
    //
    // By placing CV boundary points at (1,0) and (4,0), we
    // should obtain 3 CVs 'o', '+' and '=' as:
    //
    //
    // oooooooo+++++++++++++++
    //        o       +
    //        o       +=======
    //        o       +
    //        o       ++++++++
    //        o
    //
    // CV 0 will comprise branches 0 and 2; CV 1 branches 1, 3, 5;
    // and CV 2 branch 4. CV 0 will also cover the fork point (0,1);
    // CV 1 will cover the fork point (1, 1).

    using C = mcable;
    using testing::seq_eq;

    cable_cell cell{common_morphology::m_reg_b6, {}};
    cv_geometry geom(cell, mlocation_list{{1, 0}, {4,0}});

    EXPECT_TRUE(verify_cv_children(geom));
    ASSERT_EQ(3u, geom.size());

    mcable_list expected0 = {C{0u, 0., 1.}, C{1u, 0., 0.}, C{2u, 0., 1.}};
    EXPECT_TRUE(seq_eq(expected0, geom.cables(0)));

    mcable_list expected1 = {C{1u, 0., 1.}, C{3u, 0., 1.}, C{4u, 0., 0.}, C{5u, 0., 1.}};
    EXPECT_TRUE(seq_eq(expected1, geom.cables(1)));

    mcable_list expected2 = {C{4u, 0., 1.}};
    EXPECT_TRUE(seq_eq(expected2, geom.cables(2)));
}

TEST(cv_geom, location_cv) {
    using namespace common_morphology;

    cable_cell cell{m_reg_b6, {}};
    auto& m = cell.morphology();

    auto cv_extent = [](const cv_geometry& geom, auto cv) {
        mcable_list cl;
        util::assign(cl, geom.cables(cv));
        return mextent(cl);
    };

    auto complete = [] (const arb::morphology& m, arb::mcable c) {
        return thingify(arb::reg::complete(arb::region(c)), arb::mprovider(m)).cables();
    };

    // Two CVs per branch, plus trivial CV at forks.
    cv_geometry geom(cell,
       join(ls::on_branches(0.), ls::on_branches(0.5), ls::on_branches(1.)));

    // Confirm CVs are either trivial or a single cable covering half a branch.
    for (auto cv: geom.cell_cvs(0)) {
        auto cables = geom.cables(cv);
        if (cables.size()==1u) {
            // Half branch cable.
            mcable cable = cables.front();
            ASSERT_TRUE((cable.prox_pos==0 && cable.dist_pos==0.5 ) ||
                        (cable.prox_pos==0.5 && cable.dist_pos==1.));
        }
        else {
            // Trivial CV over fork point.
            mcable cable0 = cables.front();
            ASSERT_TRUE(cable0.prox_pos==cable0.dist_pos);

            mcable_list clist = util::assign_from(cables);
            ASSERT_TRUE(testing::cablelist_eq(complete(m, cable0), clist));
        }
    }

    // For positions strictly within a CV extent, CV preference should make no difference.
    for (auto prefer: {cv_prefer::cv_distal, cv_prefer::cv_proximal,
                       cv_prefer::cv_nonempty, cv_prefer::cv_empty}) {
        SCOPED_TRACE(prefer);
        for (auto bid: util::make_span(m.num_branches())) {
            for (double pos: {0.3, 0.7}) {
                mlocation loc{bid, pos};
                SCOPED_TRACE(loc);

                auto cv = geom.location_cv(0, loc, prefer);
                ASSERT_TRUE(cv_extent(geom, cv).intersects(loc));

                ASSERT_EQ(1u, geom.cables(cv).size());
                mcable cable = geom.cables(cv).front();
                EXPECT_TRUE(cable.branch==loc.branch);
                EXPECT_TRUE(cable.prox_pos<cable.dist_pos);
            }
        }
    }

    // For positions in the middle of a branch, we should get distal CV unless
    // CV prerence is `cv_proximal`.
    for (auto prefer: {cv_prefer::cv_distal, cv_prefer::cv_proximal,
                       cv_prefer::cv_nonempty, cv_prefer::cv_empty}) {
        SCOPED_TRACE(prefer);
        for (auto bid: util::make_span(m.num_branches())) {
            mlocation loc{bid, 0.5};
            SCOPED_TRACE(loc);

            auto cv = geom.location_cv(0, loc, prefer);
            ASSERT_TRUE(cv_extent(geom, cv).intersects(loc));

            ASSERT_EQ(1u, geom.cables(cv).size());
            mcable cable = geom.cables(cv).front();
            EXPECT_TRUE(cable.branch==loc.branch);
            EXPECT_TRUE(cable.prox_pos<cable.dist_pos);

            if (prefer==cv_prefer::cv_proximal) {
                EXPECT_EQ(0., cable.prox_pos);
            }
            else {
                EXPECT_EQ(0.5, cable.prox_pos);
            }
        }
    }

    // For the head of a non-root branch, we should get the trivial CV over the
    // fork for `cv_proximal` or `cv_empty`; otherwise the CV over the first
    // half of the branch.
    for (auto prefer: {cv_prefer::cv_distal, cv_prefer::cv_proximal,
                       cv_prefer::cv_nonempty, cv_prefer::cv_empty}) {
        SCOPED_TRACE(prefer);
        for (auto bid: util::make_span(m.num_branches())) {
            if (m.branch_parent(bid)==mnpos) continue;

            mlocation loc{bid, 0.};
            SCOPED_TRACE(loc);

            auto cv = geom.location_cv(0, loc, prefer);
            ASSERT_TRUE(cv_extent(geom, cv).intersects(loc));

            auto cables = geom.cables(cv);
            switch (prefer) {
            case cv_prefer::cv_proximal:
            case cv_prefer::cv_empty:
                EXPECT_NE(1u, cables.size());
                break;
            case cv_prefer::cv_distal:
            case cv_prefer::cv_nonempty:
                EXPECT_EQ(1u, cables.size());
                EXPECT_EQ(0.5, cables.front().dist_pos);
                break;
            }
        }
    }

    // For the tail of a non-terminal branch, we should get the trivial CV over the
    // fork for `cv_distal` or `cv_empty`; otherwise the CV over the second
    // half of the branch.
    for (auto prefer: {cv_prefer::cv_distal, cv_prefer::cv_proximal,
                       cv_prefer::cv_nonempty, cv_prefer::cv_empty}) {
        SCOPED_TRACE(prefer);
        for (auto bid: util::make_span(m.num_branches())) {
            if (m.branch_children(bid).empty()) continue;

            mlocation loc{bid, 1.};
            SCOPED_TRACE(loc);

            auto cv = geom.location_cv(0, loc, prefer);
            ASSERT_TRUE(cv_extent(geom, cv).intersects(loc));

            auto cables = geom.cables(cv);
            switch (prefer) {
            case cv_prefer::cv_proximal:
            case cv_prefer::cv_nonempty:
                EXPECT_EQ(1u, cables.size());
                EXPECT_EQ(0.5, cables.front().prox_pos);
                break;
            case cv_prefer::cv_distal:
            case cv_prefer::cv_empty:
                EXPECT_NE(1u, cables.size());
                break;
            }
        }
    }
}

TEST(cv_geom, multicell) {
    using namespace common_morphology;
    using index_type = cv_geometry::index_type;

    cable_cell cell = cable_cell(m_reg_b6, {});

    cv_geometry geom(cell, ls::on_branches(0.5));
    unsigned n_cv = geom.size();

    cv_geometry geom2 = geom;
    append(geom2, geom);

    EXPECT_TRUE(verify_cv_children(geom));

    ASSERT_EQ(2*n_cv, geom2.size());
    for (unsigned i = 0; i<n_cv; ++i) {
        EXPECT_EQ(geom.cv_parent[i], geom2.cv_parent[i]);

        if (geom2.cv_parent[i]==-1) {
            EXPECT_EQ(-1, geom2.cv_parent[i+n_cv]);
        }
        else {
            EXPECT_EQ(geom2.cv_parent[i]+(int)n_cv, geom2.cv_parent[i+n_cv]);
        }
        EXPECT_EQ(0, geom2.cv_to_cell[i]);
        EXPECT_EQ(1, geom2.cv_to_cell[i+n_cv]);

        mcable_list cables, cables1, cables2;
        util::assign(cables, geom.cables(i));
        util::assign(cables1, geom2.cables(i));
        util::assign(cables2, geom2.cables(i+n_cv));
        EXPECT_EQ(cables, cables1);
        EXPECT_EQ(cables, cables2);
    }

    EXPECT_EQ((std::pair<index_type, index_type>(0, n_cv)), geom2.cell_cv_interval(0));
    EXPECT_EQ((std::pair<index_type, index_type>(n_cv, 2*n_cv)), geom2.cell_cv_interval(1));
}

TEST(region_cv, empty) {
    using namespace common_morphology;

    cable_cell empty_cell{m_empty, {}};
    cell_cv_data cv_data(empty_cell, ls::nil());

    auto all_cv = intersect_region(reg::all(), cv_data);
    EXPECT_EQ(0u, all_cv.size());

    auto tag1_cv = intersect_region(reg::tagged(1), cv_data);
    EXPECT_EQ(0u, tag1_cv.size());
}

TEST(region_cv, trivial) {
    using namespace common_morphology;

    for (auto& p: test_morphologies) {
        if (p.second.empty()) continue;

        SCOPED_TRACE(p.first);
        cable_cell cell{p.second, {}};

        // One CV comprising whole cell:
        cell_cv_data cell_geom1(cell, ls::nil());

        auto all_cv  = intersect_region(reg::all(), cell_geom1);
        auto tag1_cv = intersect_region(reg::tagged(1), cell_geom1);
        auto tag2_cv = intersect_region(reg::tagged(2), cell_geom1);

        EXPECT_EQ(1u,  all_cv.size());
        EXPECT_EQ(0u, all_cv.front().idx);
        EXPECT_EQ(1., all_cv.front().proportion);

        EXPECT_EQ(1u,  tag1_cv.size());
        EXPECT_EQ(0u, tag1_cv.front().idx);
        EXPECT_EQ(1., tag1_cv.front().proportion);

        EXPECT_EQ(0u, tag2_cv.size());
    }
}

TEST(region_cv, custom_geometry) {
    auto almost_eq = [](const cv_proportion& a, const cv_proportion& b) {
        return a.idx==b.idx && (fabs(a.proportion-b.proportion)<1e-6);
    };

    using namespace arborio::literals;
    segment_tree tree;
    //  the test morphology:
    //
    //         _        _  y=40
    //          \       /
    //  seg3     \     /   seg2
    //  branch2   \   /    branch1
    //             \ /
    //              | y=25
    //              |
    //              | seg1
    //              | branch0
    //              |
    //              |
    //              - y=5
    //              | seg0
    //              _ y=-5

    // Root branch.
    mpoint psoma_p{0, -5, 0, 5};
    mpoint psoma_d{0,  5, 0, 5};

    msize_t ssoma = tree.append(mnpos, psoma_p, psoma_d, 1);

    mpoint py1_p{0,  5, 0, 1};
    mpoint py1_d{0, 25, 0, 1};

    msize_t sy1 = tree.append(ssoma, py1_p, py1_d, 3);

    // branch 1 of y: translation (9,15) in one segment
    mpoint py2_d{ 9, 40, 0, 1};
    tree.append(sy1, py2_d, 3);

    // branch 2 of y: translation (-9,15) in 2 segments
    mpoint py3_d{-9, 40, 0, 1};
    tree.append(sy1, py3_d, 3);

    morphology m(tree);

    label_dict l;
    l.set("all", reg::all());
    l.set("soma", reg::tagged(1));
    l.set("dend", reg::tagged(3));
    l.set("custom0", reg::cable(0, 0, 0.5));
    l.set("custom1", reg::cable(1, 0.5, 1));
    l.set("custom2", join(reg::named("custom0"), reg::named("custom1")));
    l.set("custom3", reg::cable(0, 1./6., 3./5.));
    l.set("custom4", reg::cable(0, 3./5., 1));
    {
        decor d;
        // Discretize by segment
        d.set_default(cv_policy_every_segment());
        auto cell = cable_cell(m, d, l);
        auto geom = cv_data(cell);
        EXPECT_TRUE(geom);

        auto cv0 = cv_proportion{0, 1};
        auto cv1 = cv_proportion{1, 1};
        auto cv1_quarter = cv_proportion{1, 0.25};
        auto cv2 = cv_proportion{3, 1};
        auto cv3 = cv_proportion{4, 1};
        auto cv2_half = cv_proportion{3, 0.5};

        auto all_cv = intersect_region("all"_lab, geom.value());
        EXPECT_EQ(4u,  all_cv.size());
        EXPECT_TRUE(almost_eq(cv0, all_cv[0]));
        EXPECT_TRUE(almost_eq(cv1, all_cv[1]));
        EXPECT_TRUE(almost_eq(cv2, all_cv[2]));
        EXPECT_TRUE(almost_eq(cv3, all_cv[3]));

        auto soma_cv = intersect_region("soma"_lab, geom.value());
        EXPECT_EQ(1u, soma_cv.size());
        EXPECT_TRUE(almost_eq(cv0, soma_cv[0]));

        auto dend_cv = intersect_region("dend"_lab, geom.value());
        EXPECT_EQ(3u, dend_cv.size());
        EXPECT_TRUE(almost_eq(cv1, dend_cv[0]));
        EXPECT_TRUE(almost_eq(cv2, dend_cv[1]));
        EXPECT_TRUE(almost_eq(cv3, dend_cv[2]));

        auto c0_cv = intersect_region("custom0"_lab, geom.value());
        EXPECT_EQ(2u, c0_cv.size());
        EXPECT_TRUE(almost_eq(cv0, c0_cv[0]));
        EXPECT_TRUE(almost_eq(cv1_quarter, c0_cv[1]));

        auto c1_cv = intersect_region("custom1"_lab, geom.value());
        EXPECT_EQ(1u, c1_cv.size());
        EXPECT_TRUE(almost_eq(cv2_half, c1_cv[0]));

        auto c2_cv = intersect_region("custom2"_lab, geom.value());
        EXPECT_EQ(3u, c2_cv.size());
        EXPECT_TRUE(almost_eq(cv0, c2_cv[0]));
        EXPECT_TRUE(almost_eq(cv1_quarter, c2_cv[1]));
        EXPECT_TRUE(almost_eq(cv2_half, c2_cv[2]));
    }
    {
        decor d;
        // Discretize using explicit locset
        auto ls = locset({
          {0, 0},
          {0, 0.1},
          {0, 0.5},
          {0, 0.7},
          {0, 0.9},
          {1, 0.1},
          {1, 1},
          {2, 0.2},
          {2, 1}
        });
        d.set_default(cv_policy_explicit(ls));
        auto cell = cable_cell(m, d, l);
        auto geom = cv_data(cell);
        EXPECT_TRUE(geom);

        auto cv0 = cv_proportion{0, 1};
        auto cv1 = cv_proportion{1, 1};
        auto cv1_part0 = cv_proportion{1, 7./8.};
        auto cv1_part1 = cv_proportion{1, 1./8.};
        auto cv1_part2 = cv_proportion{1, 3./4.};
        auto cv2 = cv_proportion{2, 1};
        auto cv2_part = cv_proportion{2, 1./2.};
        auto cv3 = cv_proportion{3, 1};
        auto cv4 = cv_proportion{4, 1};
        auto cv4_part = cv_proportion{4, 10/(sqrt(306)+10)};
        auto cv5 = cv_proportion{5, 1};
        auto cv5_part = cv_proportion{5, 15./27.};
        auto cv6 = cv_proportion{6, 1};

        auto all_cv = intersect_region("all"_lab, geom.value());
        EXPECT_EQ(7u,  all_cv.size());
        EXPECT_TRUE(almost_eq(cv0, all_cv[0]));
        EXPECT_TRUE(almost_eq(cv1, all_cv[1]));
        EXPECT_TRUE(almost_eq(cv2, all_cv[2]));
        EXPECT_TRUE(almost_eq(cv3, all_cv[3]));
        EXPECT_TRUE(almost_eq(cv4, all_cv[4]));
        EXPECT_TRUE(almost_eq(cv5, all_cv[5]));
        EXPECT_TRUE(almost_eq(cv6, all_cv[6]));

        auto soma_cv = intersect_region("soma"_lab, geom.value());
        EXPECT_EQ(2u, soma_cv.size());
        EXPECT_TRUE(almost_eq(cv0, soma_cv[0]));
        EXPECT_TRUE(almost_eq(cv1_part0, soma_cv[1]));

        auto dend_cv = intersect_region("dend"_lab, geom.value());
        EXPECT_EQ(6u, dend_cv.size());
        EXPECT_TRUE(almost_eq(cv1_part1, dend_cv[0]));
        EXPECT_TRUE(almost_eq(cv2, dend_cv[1]));
        EXPECT_TRUE(almost_eq(cv3, dend_cv[2]));
        EXPECT_TRUE(almost_eq(cv4, dend_cv[3]));
        EXPECT_TRUE(almost_eq(cv5, dend_cv[4]));
        EXPECT_TRUE(almost_eq(cv6, dend_cv[5]));
        EXPECT_TRUE(almost_eq(cv6, dend_cv[5]));

        auto c2_cv = intersect_region("custom2"_lab, geom.value());
        EXPECT_EQ(3u, c2_cv.size());
        EXPECT_TRUE(almost_eq(cv0, c2_cv[0]));
        EXPECT_TRUE(almost_eq(cv1, c2_cv[1]));
        EXPECT_TRUE(almost_eq(cv5_part, c2_cv[2]));

        auto c3_cv = intersect_region("custom3"_lab, geom.value());
        EXPECT_EQ(2u, c3_cv.size());
        EXPECT_TRUE(almost_eq(cv1_part2, c3_cv[0]));
        EXPECT_TRUE(almost_eq(cv2_part, c3_cv[1]));

        auto c4_cv = intersect_region("custom4"_lab, geom.value());
        EXPECT_EQ(3u, c2_cv.size());
        EXPECT_TRUE(almost_eq(cv2_part, c4_cv[0]));
        EXPECT_TRUE(almost_eq(cv3, c4_cv[1]));
        EXPECT_TRUE(almost_eq(cv4_part, c4_cv[2]));
    }
}
