#include <algorithm>
#include <utility>

#include <arbor/util/optional.hpp>
#include <arbor/cable_cell.hpp>
#include <arbor/morph/morphology.hpp>
#include <arbor/morph/locset.hpp>

#include "fvm_layout.hpp"

#include "common.hpp"
#include "common_morphologies.hpp"
#include "../common_cells.hpp"

using namespace arb;
using util::make_span;

template <typename... A>
locset as_locset(mlocation head, A... tail) {
    return join(ls::location(head), ls::location(tail)...);
}

TEST(cv_geom, empty) {
    using namespace common_morphology;

    cable_cell empty_cell{m_empty};
    cv_geometry geom = cv_geometry_from_ends(empty_cell, ls::nil());

    EXPECT_TRUE(geom.cv_parent.empty());
    EXPECT_TRUE(geom.cv_cables.empty());
    EXPECT_TRUE(geom.cv_cables_divs.empty());

    EXPECT_EQ(0u, geom.size()); // size()/empty() reflects number of CVs.
    EXPECT_EQ(1u, geom.n_cell()); // can have no CVs but >0 cells.
}

TEST(cv_geom, trivial) {
    using namespace common_morphology;

    for (auto& p: test_morphologies) {
        if (p.second.empty()) continue;

        SCOPED_TRACE(p.first);
        cable_cell cell{p.second};
        auto& m = cell.morphology();


        // Equivalent ways of specifying one CV comprising whole cell:
        cv_geometry geom1 = cv_geometry_from_ends(cell, ls::nil());
        cv_geometry geom2 = cv_geometry_from_ends(cell, ls::terminal());

        EXPECT_EQ(1u, geom1.size());
        EXPECT_EQ(geom1.cv_cables, geom2.cv_cables);

        // These are equivalent too, if there is a single root branch.
        cv_geometry geom3 = cv_geometry_from_ends(cell, ls::root());
        cv_geometry geom4 = cv_geometry_from_ends(cell, join(ls::root(), ls::terminal()));

        EXPECT_EQ(geom3.cv_cables, geom4.cv_cables);
        if (m.branch_children(mnpos).size()==1) {
            EXPECT_EQ(geom1.cv_cables, geom4.cv_cables);
        }

        mcable_list all_cables = thingify(reg::all(), cell.provider());
        EXPECT_TRUE(testing::seq_eq(all_cables, geom1.cables(0)));
    }
}

TEST(cv_geom, one_cv_per_branch) {
    using namespace common_morphology;

    for (auto& p: test_morphologies) {
        if (p.second.empty()) continue;
        SCOPED_TRACE(p.first);

        cable_cell cell{p.second};
        auto& m = cell.morphology();

        cv_geometry geom = cv_geometry_from_ends(cell, ls::on_branches(0));

        // Expect trivial CVs at every fork point, and single-cable CVs for each branch.
        std::vector<unsigned> seen_branches(m.num_branches(), 0);
        auto n_branch_child = [&m](msize_t b) { return m.branch_children(b).size(); };
        for (auto i: make_span(geom.size())) {
            auto cables = geom.cables(i);

            ASSERT_EQ(1u, cables.size());
            auto c = cables.front();

            if (c.prox_pos==c.dist_pos) {
                if (c.branch==0 && c.prox_pos==0) {
                    EXPECT_TRUE(n_branch_child(mnpos)>1);
                }
                else {
                    EXPECT_EQ(1., c.prox_pos);
                    EXPECT_TRUE(n_branch_child(c.branch)>1);
                }
            }
            else {
                ++seen_branches[c.branch];
                EXPECT_EQ(1., seen_branches[c.branch]);
                EXPECT_EQ(0., c.prox_pos);
                EXPECT_EQ(1., c.dist_pos);

                // Confirm parent CV is fork CV:
                if (i>0) {
                    mlocation pfork = canonical(m, mlocation{c.branch, 0.});

                    auto pcables = geom.cables(geom.cv_parent[i]);
                    ASSERT_EQ(1u, pcables.size());

                    mcable p = pcables.front();
                    EXPECT_EQ(pfork.branch, p.branch);
                    EXPECT_EQ(p.prox_pos, p.dist_pos);

                    if (p.branch==0) {
                        EXPECT_TRUE(p.prox_pos==0 || p.prox_pos==1);
                    }
                    else {
                        EXPECT_EQ(1., p.prox_pos);
                    }
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

        cable_cell cell{p.second};
        auto& m = cell.morphology();

        cv_geometry geom = cv_geometry_from_ends(cell, ls::on_branches(0.5));

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
    // and CV 2 branch 4.

    using L = mlocation;
    using C = mcable;
    using testing::seq_eq;

    cable_cell cell{common_morphology::m_reg_b6};
    cv_geometry geom = cv_geometry_from_ends(cell, as_locset(L{1, 0}, L{4,0}));

    ASSERT_EQ(3u, geom.size());

    mcable_list expected0 = {C{0u, 0., 1.}, C{2u, 0., 1.}};
    EXPECT_TRUE(seq_eq(expected0, geom.cables(0)));

    mcable_list expected1 = {C{1u, 0., 1.}, C{3u, 0., 1.}, C{5u, 0., 1.}};
    EXPECT_TRUE(seq_eq(expected1, geom.cables(1)));

    mcable_list expected2 = {C{4u, 0., 1.}};
    EXPECT_TRUE(seq_eq(expected2, geom.cables(2)));
}

TEST(cv_geom, location_cv) {
    using namespace common_morphology;

    cable_cell cell{m_reg_b6};

    // Two CVs per branch, plus trivial CV at forks.
    cv_geometry geom = cv_geometry_from_ends(cell,
       join(ls::on_branches(0.0), ls::on_branches(0.5), ls::on_branches(1.)));

    // Confirm CVs are all only one cable, and either trivial or half a branch.
    for (auto cv: geom.cell_cvs(0)) {
        auto cables = geom.cables(cv);
        ASSERT_EQ(1u, cables.size());

        mcable cable = cables.front();
        ASSERT_TRUE(cable.prox_pos==cable.dist_pos ||
                    (cable.prox_pos==0 && cable.dist_pos==0.5 )||
                    (cable.prox_pos==0.5 && cable.dist_pos==1.));
    }

    // Where ambiguous, CV found should be one with a non-trivial cable containing the location.
    for (auto bid: util::make_span(cell.morphology().num_branches())) {
        for (double pos: {0., 0.3, 0.5, 0.7, 1.}) {
            mlocation loc{bid, pos};
            SCOPED_TRACE(loc);

            auto cv = geom.location_cv(0, loc);

            mcable cable = geom.cables(cv).front();
            EXPECT_TRUE(cable.prox_pos<cable.dist_pos);
            EXPECT_TRUE(cable.prox_pos<=pos);
            EXPECT_TRUE(cable.dist_pos>=pos);
            EXPECT_TRUE(cable.branch==loc.branch);
        }
    }

    // CV 0 will comprise branches 0 and 2; CV 1 branches 1, 3, 5;
    // and CV 2 branch 4 (see test cv_geom.weird above).
    cv_geometry gweird = cv_geometry_from_ends(cell, as_locset(mlocation{1, 0}, mlocation{4,0}));

    // Expect location (4, 0.) to be in CV 2, but colocated locations
    // (3, 0.), (5, 0.) and (1, 1.) to be in CV 1.
    EXPECT_EQ(2u, gweird.location_cv(0, mlocation{4, 0.}));
    EXPECT_EQ(1u, gweird.location_cv(0, mlocation{3, 0.}));
    EXPECT_EQ(1u, gweird.location_cv(0, mlocation{5, 0.}));
    EXPECT_EQ(1u, gweird.location_cv(0, mlocation{1, 1.}));
}
