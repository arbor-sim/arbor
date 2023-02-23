#include <fstream>
#include <cmath>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <arbor/cable_cell.hpp>
#include <arbor/morph/morphexcept.hpp>
#include <arbor/morph/morphology.hpp>
#include <arbor/morph/primitives.hpp>
#include <arbor/morph/segment_tree.hpp>

#include <arborio/swcio.hpp>

#include "util/span.hpp"

#include "morph_pred.hpp"

// Test basic functions on properties of mpoint
TEST(morphology, mpoint) {
    using mp = arb::mpoint;

    EXPECT_EQ(arb::distance(mp{0,0,0},mp{0   , 0   , 0   }), 0.);
    EXPECT_EQ(arb::distance(mp{0,0,0},mp{3.14, 0   , 0   }), 3.14);
    EXPECT_EQ(arb::distance(mp{0,0,0},mp{0   , 3.14, 0   }), 3.14);
    EXPECT_EQ(arb::distance(mp{0,0,0},mp{0   , 0   , 3.14}), 3.14);
    EXPECT_EQ(arb::distance(mp{0,0,0},mp{1   , 2   , 3   }), std::sqrt(14.));

    EXPECT_TRUE(arb::is_collocated(mp{0,0,0}, mp{0,0,0}));
    EXPECT_TRUE(arb::is_collocated(mp{3,0,0}, mp{3,0,0}));
    EXPECT_TRUE(arb::is_collocated(mp{0,3,0}, mp{0,3,0}));
    EXPECT_TRUE(arb::is_collocated(mp{0,0,3}, mp{0,0,3}));
    EXPECT_TRUE(arb::is_collocated(mp{2,0,3}, mp{2,0,3}));

    EXPECT_FALSE(arb::is_collocated(mp{1,0,3}, mp{2,0,3}));
    EXPECT_FALSE(arb::is_collocated(mp{2,1,3}, mp{2,0,3}));
    EXPECT_FALSE(arb::is_collocated(mp{2,0,1}, mp{2,0,3}));
}

// For different parent index vectors, attempt multiple valid and invalid sample sets.
TEST(morphology, construction) {
    constexpr auto npos = arb::mnpos;
    using arb::util::make_span;
    using mp = arb::mpoint;
    using pvec = std::vector<arb::msize_t>;
    {
        pvec p = {npos, 0};
        std::vector<mp> s = {
            {0.0, 0.0, 0.0, 1.0},
            {0.0, 0.0, 1.0, 1.0}};

        arb::segment_tree tree;
        tree.append(npos, s[0], s[1], 1);
        auto m = arb::morphology(tree);

        EXPECT_EQ(1u, m.num_branches());
    }
    {
        pvec p = {npos, 0, 1};
        // 2-segment cable
        std::vector<mp> s = {
            {0.0, 0.0, 0.0, 5.0},
            {0.0, 0.0, 1.0, 1.0},
            {0.0, 0.0, 8.0, 1.0} };

        arb::segment_tree sm;
        sm.append(npos, s[0], s[1], 2);
        sm.append(   0, s[1], s[2], 2);
        auto m = arb::morphology(sm);

        EXPECT_EQ(1u, m.num_branches());
    }
    {
        //              0       |
        //            1   3     |
        //          2           |
        pvec p = {npos, 0, 1, 0};
        // two cables: 1x2 segments, 1x1 segment.
        std::vector<mp> s = {
            {0.0, 0.0, 0.0, 5.0},
            {0.0, 0.0, 5.0, 1.0},
            {0.0, 0.0, 6.0, 1.0},
            {0.0, 4.0, 0.0, 1.0}};

        arb::segment_tree tree;
        tree.append(npos, s[0], s[1], 1);
        tree.append(   0, s[1], s[2], 2);
        tree.append(npos, s[0], s[3], 1);
        auto m = arb::morphology(tree);

        EXPECT_EQ(2u, m.num_branches());
    }
    {
        //              0       |
        //            1   3     |
        //          2       4    |

        // two cables: 2 segments each
        std::vector<mp> s = {
            {0.0, 0.0, 0.0, 5.0},
            {0.0, 0.0, 5.0, 1.0},
            {0.0, 0.0, 8.0, 1.0},
            {0.0, 5.0, 0.0, 1.0},
            {0.0, 8.0, 0.0, 1.0}};

        arb::segment_tree tree;
        tree.append(npos, s[0], s[1], 1);
        tree.append(   0, s[1], s[2], 1);
        tree.append(npos, s[0], s[3], 1);
        tree.append(   2, s[3], s[4], 1);
        auto m = arb::morphology(tree);

        EXPECT_EQ(2u, m.num_branches());
    }
    {
        //              0       |
        //            1   3     |
        //          2    4  5   |
        pvec p = {npos, 0, 1, 0};
        // 4 cables
        std::vector<mp> s = {
            {0.0, 0.0, 0.0, 5.0},
            {0.0, 0.0, 5.0, 1.0},
            {0.0, 0.0, 6.0, 1.0},
            {0.0, 4.0, 0.0, 1.0},
            {0.0, 4.0, 1.0, 1.0},
            {0.0, 4.0, 2.0, 1.0}};

        arb::segment_tree tree;
        tree.append(npos, s[0], s[1], 1);
        tree.append(   0, s[1], s[2], 2);
        tree.append(npos, s[0], s[3], 1);
        tree.append(   2, s[3], s[4], 1);
        tree.append(   2, s[4], s[5], 1);
        auto m = arb::morphology(tree);

        EXPECT_EQ(4u, m.num_branches());
    }
}

// test that morphology generates branch child-parent structure correctly.
TEST(morphology, branches) {
    using pvec = std::vector<arb::msize_t>;
    using svec = std::vector<arb::mpoint>;
    auto npos = arb::mnpos;

    auto check_terminal_branches = [](const arb::morphology& m) {
        pvec expected;
        arb::msize_t n = m.num_branches();

        for (arb::msize_t i = 0; i<n; ++i) {
            if (m.branch_children(i).empty()) expected.push_back(i);
        }
        EXPECT_EQ(expected, m.terminal_branches());
    };

    {
        // 0 - 1
        svec p = {
            {0.,0.,0.,3.},
            {10.,0.,0.,3.},
        };
        arb::segment_tree tree;
        tree.append(npos, p[0], p[1], 1);
        arb::morphology m(tree);

        EXPECT_EQ(1u, m.num_branches());
        EXPECT_EQ(npos, m.branch_parent(0));
        EXPECT_EQ(pvec{}, m.branch_children(0));

        check_terminal_branches(m);
    }
    {
        // 0 - 1 - 2

        // the morphology is a single unbranched cable.
        svec p = {
            {0.,0.,0.,3.},
            {10.,0.,0.,3.},
            {100,0,0,3},
        };
        arb::segment_tree tree;
        tree.append(npos, p[0], p[1], 1);
        tree.append(   0, p[1], p[2], 1);
        arb::morphology m(tree);

        EXPECT_EQ(1u, m.num_branches());
        EXPECT_EQ(npos, m.branch_parent(0));
        EXPECT_EQ(pvec{}, m.branch_children(0));
        EXPECT_EQ(pvec{0}, m.terminal_branches());

        check_terminal_branches(m);
    }
    {
        // 6 segemnts and six branches.
        // A single branch at the root that bifurcates, and a further 3 branches
        // attached to the first bifurcation.
        //       0      |
        //      / \     |
        //     1   2    |
        //   / | \      |
        //  3  4  5     |

        svec p = {
            {0., 0.,0.,3.},
            {1., 0.,0.,3.},
            {2., 1.,0.,3.},
            {2.,-1.,0.,3.},
            {3., 2.,0.,3.},
            {3., 1.,0.,3.},
            {3., 0.,0.,3.},
        };
        arb::segment_tree tree;
        tree.append(npos, p[0], p[1], 1);
        tree.append(   0, p[1], p[2], 1);
        tree.append(   0, p[1], p[3], 1);
        tree.append(   1, p[2], p[4], 1);
        tree.append(   1, p[2], p[5], 1);
        tree.append(   1, p[2], p[6], 1);
        arb::morphology m(tree);

        EXPECT_EQ(6u, m.num_branches());
        EXPECT_EQ(npos, m.branch_parent(0));
        EXPECT_EQ(  0u, m.branch_parent(1));
        EXPECT_EQ(  0u, m.branch_parent(2));
        EXPECT_EQ(  1u, m.branch_parent(3));
        EXPECT_EQ(  1u, m.branch_parent(4));
        EXPECT_EQ(  1u, m.branch_parent(5));
        EXPECT_EQ((pvec{1,2}), m.branch_children(0));
        EXPECT_EQ((pvec{3,4,5}), m.branch_children(1));
        EXPECT_EQ((pvec{}), m.branch_children(2));
        EXPECT_EQ((pvec{}), m.branch_children(3));
        EXPECT_EQ((pvec{}), m.branch_children(4));
        EXPECT_EQ((pvec{}), m.branch_children(5));
        EXPECT_EQ((pvec{2,3,4,5}), m.terminal_branches());

        check_terminal_branches(m);
    }
    {
        //     0      |
        //    / \     |
        //   1   2    |

        svec p = {
            { 0, 0,0, 5},
            {10, 0,0, 5},
            { 0,10,0, 5},
        };
        arb::segment_tree tree;
        tree.append(npos, p[0], p[1], 3);
        tree.append(npos, p[0], p[2], 3);
        arb::morphology m(tree);

        EXPECT_EQ(2u, m.num_branches());
        EXPECT_EQ(npos, m.branch_parent(0));
        EXPECT_EQ(npos,   m.branch_parent(1));
        EXPECT_EQ(pvec{}, m.branch_children(0));
        EXPECT_EQ(pvec{},  m.branch_children(1));
        EXPECT_EQ((pvec{0,1}), m.terminal_branches());

        check_terminal_branches(m);
    }
    {
        // 7 segments, 4 branches
        //
        //              0           |
        //             / \          |
        //            1   3         |
        //           /     \        |
        //          2       4       |
        //                 / \      |
        //                5   6     |
        //                     \    |
        //                      7   |
        svec p = {
            {  0,  0,  0, 10},
            { 10,  0,  0,  2},
            {100,  0,  0,  2},
            {  0, 10,  0,  2},
            {  0,100,  0,  2},
            {100,100,  0,  2},
            {  0,200,  0,  2},
            {  0,300,  0,  2},
        };
        arb::segment_tree tree;
        tree.append(npos, p[0], p[1], 1);
        tree.append(   0, p[1], p[2], 1);
        tree.append(npos, p[0], p[3], 1);
        tree.append(   2, p[3], p[4], 1);
        tree.append(   3, p[4], p[5], 1);
        tree.append(   3, p[4], p[6], 1);
        tree.append(   5, p[6], p[7], 1);
        arb::morphology m(tree);

        EXPECT_EQ(4u, m.num_branches());
        EXPECT_EQ(npos, m.branch_parent(0));
        EXPECT_EQ(npos, m.branch_parent(1));
        EXPECT_EQ(1u,   m.branch_parent(2));
        EXPECT_EQ(1u,   m.branch_parent(3));
        EXPECT_EQ((pvec{}),    m.branch_children(0));
        EXPECT_EQ((pvec{2,3}), m.branch_children(1));
        EXPECT_EQ((pvec{}),    m.branch_children(2));
        EXPECT_EQ((pvec{}),    m.branch_children(3));

        check_terminal_branches(m);
    }
}

// hipcc bug in reading DATADIR
#ifndef ARB_HIP
TEST(morphology, swc) {
    std::string datadir{DATADIR};
    auto fname = datadir + "/pyramidal.swc";
    std::ifstream fid(fname);
    if (!fid.is_open()) {
        std::cerr << "unable to open file " << fname << "... skipping test\n";
        return;
    }

    // Load swc samples from file.
    auto swc = arborio::parse_swc(fid);

    // Build a segmewnt_tree from swc samples.
    auto m = arborio::load_swc_arbor(swc);
    EXPECT_EQ(221u, m.num_branches()); // 219 branches + 2 from divided soma.
}
#endif

arb::morphology make_4_branch_morph() {
    using svec = std::vector<arb::mpoint>;
    constexpr auto npos = arb::mnpos;

    // Eight points, 7 segments, 3 branches.
    //          sample   branch
    //            0         0
    //           1 3       0 1
    //          2   4     0   1
    //             5 6       2 3
    //                7         3
    svec p = {
        {  0,  0,  0,  2},
        { 10,  0,  0,  2},
        {100,  0,  0,  2},
        {  0, 10,  0,  2},
        {  0,100,  0,  2},
        {100,100,  0,  2},
        {  0,200,  0,  2},
        {  0,300,  0,  2},
    };
    arb::segment_tree tree;
    tree.append(npos, p[0], p[1], 1);
    tree.append(   0, p[1], p[2], 1);
    tree.append(npos, p[0], p[3], 1);
    tree.append(   2, p[3], p[4], 1);
    tree.append(   3, p[4], p[5], 1);
    tree.append(   3, p[4], p[6], 1);
    tree.append(   4, p[6], p[7], 1);

    return arb::morphology(tree);
}

TEST(morphology, minset) {
    auto m = make_4_branch_morph();

    using ll = arb::mlocation_list;

    EXPECT_EQ((ll{}), minset(m, ll{}));
    EXPECT_EQ((ll{{2,0.1}}), minset(m, ll{{2,0.1}}));
    EXPECT_EQ((ll{{0,0.5}, {1,0.5}}), minset(m, ll{{0,0.5}, {1,0.5}}));
    EXPECT_EQ((ll{{0,0.5}}), minset(m, ll{{0,0.5}}));
    EXPECT_EQ((ll{{0,0}, {1,0}}), minset(m, ll{{0,0}, {0,0.5}, {1,0}, {1,0.5}}));
    EXPECT_EQ((ll{{0,0}, {1,0.5}}), minset(m, ll{{0,0}, {0,0.5}, {1,0.5}, {2,0.5}}));
    EXPECT_EQ((ll{{0,0}, {2,0.5}}), minset(m, ll{{0,0}, {0,0.5}, {2,0.5}}));
    EXPECT_EQ((ll{{0,0}, {2,0.5}, {3,0}}), minset(m, ll{{0,0}, {0,0.5}, {2,0.5}, {3,0}, {3,1}}));
    EXPECT_EQ((ll{{0,0}, {1,0}}), minset(m, ll{{0,0}, {0,0.5}, {1,0}, {2,0.5}, {3,0}, {3,1}}));
}

TEST(morphology, maxset) {
    auto m = make_4_branch_morph();

    using ll = arb::mlocation_list;

    EXPECT_EQ((ll{}), maxset(m, ll{}));
    EXPECT_EQ((ll{{2,0.1}}), maxset(m, ll{{2,0.1}}));
    EXPECT_EQ((ll{{0,0.5}, {1,0.5}}), maxset(m, ll{{0,0.5}, {1,0.5}}));
    EXPECT_EQ((ll{{0,0.5}}), maxset(m, ll{{0,0.5}}));
    EXPECT_EQ((ll{{0,0.5}, {1,0.5}}), maxset(m, ll{{0,0}, {0,0.5}, {1,0}, {1,0.5}}));
    EXPECT_EQ((ll{{0,0.5}, {2,0.5}}), maxset(m, ll{{0,0}, {0,0.5}, {1,0.5}, {2,0.5}}));
    EXPECT_EQ((ll{{0,0.5}, {2,0.5}}), maxset(m, ll{{0,0}, {0,0.5}, {2,0.5}}));
    EXPECT_EQ((ll{{0,0.5}, {2,0.5}, {3,1}}), maxset(m, ll{{0,0}, {0,0.5}, {2,0.5}, {3,0}, {3,1}}));
    EXPECT_EQ((ll{{2,0.5}, {3,0.5}}), maxset(m, ll{{1,0.5}, {2,0.5}, {3,0}, {3,0.2}, {3,0.5}}));
}

// Testing mextent; intersection and join operations are
// exercised by region/locset thingifies in test_morph_expr.cpp.

TEST(mextent, invariants) {
    using namespace arb;
    using testing::cablelist_eq;

    auto m = make_4_branch_morph();

    using cl = mcable_list;

    mextent x1(cl{{1, 0.25, 0.75}});
    ASSERT_TRUE(x1.test_invariants(m));
    EXPECT_TRUE(cablelist_eq(cl{{1, 0.25, 0.75}}, x1.cables()));

    mextent x2(cl{{1, 0., 0.75}});
    ASSERT_TRUE(x2.test_invariants(m));
    EXPECT_TRUE(cablelist_eq(cl{{1, 0., 0.75}}, x2.cables()));

    mextent x3(cl{{2, 0., 1.}});
    ASSERT_TRUE(x3.test_invariants(m));
    EXPECT_TRUE(cablelist_eq(cl{{2, 0., 1.}}, x3.cables()));

    // Test that overlapping cables are merged on construction.
    mextent x4(cl{{0, 0.0, 1.0}, {0, 0.5, 0.7},
                  {1, 0.2, 0.8}, {1, 0.4, 0.6},
                  {2, 0.2, 0.5}, {2, 0.5, 0.6},
                  {3, 0.2, 0.5}, {3, 0.6, 0.7}});
    ASSERT_TRUE(x4.test_invariants(m));
    EXPECT_TRUE(cablelist_eq(cl{{0, 0.0, 1.0},
                                {1, 0.2, 0.8},
                                {2, 0.2, 0.6},
                                {3, 0.2, 0.5}, {3, 0.6, 0.7}}, x4.cables()));
}

TEST(mextent, intersects) {
    using namespace arb;
    using testing::cablelist_eq;

    auto m = make_4_branch_morph();

    using cl = mcable_list;

    mextent x1(cl{{1, 0.25, 0.75}});
    EXPECT_TRUE(x1.intersects(mlocation{1, 0.25}));
    EXPECT_TRUE(x1.intersects(mlocation{1, 0.5}));
    EXPECT_TRUE(x1.intersects(mlocation{1, 0.75}));
    EXPECT_FALSE(x1.intersects(mlocation{1, 0.}));
    EXPECT_FALSE(x1.intersects(mlocation{1, 1.}));

    EXPECT_FALSE(x1.intersects(mcable{1, 0., 0.2}));
    EXPECT_TRUE(x1.intersects(mcable{1, 0., 0.25}));
    EXPECT_TRUE(x1.intersects(mcable{1, 0.4, 0.6}));
    EXPECT_TRUE(x1.intersects(mcable{1, 0.2, 0.8}));
    EXPECT_TRUE(x1.intersects(mcable{1, 0.75, 1.0}));
    EXPECT_FALSE(x1.intersects(mcable{1, 0.8, 1.0}));

    mextent x2(cl{{1, 0., 0.75}});
    EXPECT_TRUE(x2.intersects(mlocation{1, 0.}));
    EXPECT_FALSE(x2.intersects(mlocation{0, 0.}));
    EXPECT_FALSE(x2.intersects(mlocation{0, 1.}));

    mextent x3(cl{{2, 0., 1.}});
    EXPECT_FALSE(x3.intersects(mlocation{1, 0.}));
    EXPECT_FALSE(x3.intersects(mlocation{1, 1.}));
    EXPECT_FALSE(x3.intersects(mlocation{3, 0.}));
    EXPECT_FALSE(x3.intersects(mlocation{3, 1.}));
}
