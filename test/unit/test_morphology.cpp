#include <fstream>
#include <cmath>
#include <string>
#include <vector>

#include "../test/gtest.h"

#include <arbor/morph/morphexcept.hpp>
#include <arbor/morph/morphology.hpp>
#include <arbor/morph/sample_tree.hpp>
#include <arbor/cable_cell.hpp>

#include "morph/mbranch.hpp"
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

TEST(morphology, point_props) {
    arb::point_prop p = arb::point_prop_mask_none;

    EXPECT_FALSE(arb::is_terminal(p));
    EXPECT_FALSE(arb::is_fork(p));
    EXPECT_FALSE(arb::is_root(p));
    EXPECT_FALSE(arb::is_collocated(p));

    arb::set_root(p);
    EXPECT_FALSE(arb::is_terminal(p));
    EXPECT_FALSE(arb::is_fork(p));
    EXPECT_TRUE(arb::is_root(p));
    EXPECT_FALSE(arb::is_collocated(p));

    arb::set_terminal(p);
    EXPECT_TRUE(arb::is_terminal(p));
    EXPECT_FALSE(arb::is_fork(p));
    EXPECT_TRUE(arb::is_root(p));
    EXPECT_FALSE(arb::is_collocated(p));

    arb::unset_root(p);
    EXPECT_TRUE(arb::is_terminal(p));
    EXPECT_FALSE(arb::is_fork(p));
    EXPECT_FALSE(arb::is_root(p));
    EXPECT_FALSE(arb::is_collocated(p));

    arb::set_collocated(p);
    EXPECT_TRUE(arb::is_terminal(p));
    EXPECT_FALSE(arb::is_fork(p));
    EXPECT_FALSE(arb::is_root(p));
    EXPECT_TRUE(arb::is_collocated(p));

    arb::set_fork(p);
    EXPECT_TRUE(arb::is_terminal(p));
    EXPECT_TRUE(arb::is_fork(p));
    EXPECT_FALSE(arb::is_root(p));
    EXPECT_TRUE(arb::is_collocated(p));

    arb::unset_fork(p);
    arb::unset_terminal(p);
    arb::unset_collocated(p);
    EXPECT_FALSE(arb::is_terminal(p));
    EXPECT_FALSE(arb::is_fork(p));
    EXPECT_FALSE(arb::is_root(p));
    EXPECT_FALSE(arb::is_collocated(p));
}

// TODO: test sample_tree marks properties correctly

// Test internal function that parses a parent list, and marks
// each node as either root, sequential, fork or terminal.
TEST(sample_tree, properties) {
    const auto npos = arb::mnpos;
    using arb::sample_tree;
    using pp = arb::point_prop;
    using pvec = std::vector<arb::msize_t>;

    pp c = arb::point_prop_mask_collocated;
    pp r = arb::point_prop_mask_root;
    pp t = arb::point_prop_mask_terminal;
    pp s = arb::point_prop_mask_none;
    pp f = arb::point_prop_mask_fork;
    pp tc = t+c;
    pp sc = s+c;
    pp fc = f+c;

    // make a sample tree from a parent vector with non-collocated points.
    auto make_tree = [] (const pvec& parents) {
        sample_tree st;
        for (auto p: parents) st.append(p, {{0.,0.,double(st.size()),1.}, 1});
        return st;
    };
    // make a sample tree from a parent vector with collocated points.
    auto make_colloc_tree = [] (const pvec& parents) {
        sample_tree st;
        for (auto p: parents) st.append(p, {{0.,0.,0.,1.}, 1});
        return st;
    };

    {
        EXPECT_EQ(make_tree({npos}).properties(), std::vector<pp>{r});
        EXPECT_EQ(make_colloc_tree({npos}).properties(), std::vector<pp>{r});
    }

    {
        EXPECT_EQ(make_tree({npos,0}).properties(), (std::vector<pp>{r,t}));
        EXPECT_EQ(make_colloc_tree({npos,0}).properties(), (std::vector<pp>{r,tc}));
    }

    {
        EXPECT_EQ(make_tree({npos,0,1,2}).properties(), (std::vector<pp>{r,s,s,t}));
        EXPECT_EQ(make_colloc_tree({npos,0,1,2}).properties(), (std::vector<pp>{r,sc,sc,tc}));
    }

    {
        EXPECT_EQ(make_tree({npos,0,1,2,0,4,5}).properties(), (std::vector<pp>{r,s,s,t,s,s,t}));
        EXPECT_EQ(make_colloc_tree({npos,0,1,2,0,4,5}).properties(), (std::vector<pp>{r,sc,sc,tc,sc,sc,tc}));
    }

    {
        EXPECT_EQ(make_tree({npos,0,1,2,3,2,4,4,7}).properties(), (std::vector<pp>{r,s,f,s,f,t,t,s,t}));
        EXPECT_EQ(make_colloc_tree({npos,0,1,2,3,2,4,4,7}).properties(), (std::vector<pp>{r,sc,fc,sc,fc,tc,tc,sc,tc}));
    }
}

TEST(morphology, branches_from_parent_index) {
    const auto npos = arb::mnpos;
    using pvec = std::vector<arb::msize_t>;
    using mb = arb::impl::mbranch;

    // make a sample tree from a parent vector with non-collocated points.
    auto make_tree = [] (const pvec& parents) {
        arb::sample_tree st;
        for (auto p: parents) st.append(p, {{0.,0.,double(st.size()),1.}, 1});
        return st;
    };

    {   // single sample: can only be used to build a morphology with one spherical branch
        pvec parents{npos};
        auto tree = make_tree(parents);
        auto bc = arb::impl::branches_from_parent_index(parents, tree.properties(), true);
        EXPECT_EQ(1u, bc.size());
        EXPECT_EQ(mb({0}, npos), bc[0]);

        // A cable morphology can't be constructed from a single sample.
        EXPECT_THROW(arb::impl::branches_from_parent_index(parents, tree.properties(), false),
                     arb::incomplete_branch);
    }
    {
        pvec parents = {npos, 0};
        auto tree = make_tree(parents);
        auto bc = arb::impl::branches_from_parent_index(parents, tree.properties(), false);
        EXPECT_EQ(1u, bc.size());
        EXPECT_EQ(mb({0,1}, npos), bc[0]);

        // A morphology can't be constructed with a spherical soma from two samples.
        EXPECT_THROW(arb::impl::branches_from_parent_index(parents, tree.properties(), true),
                     arb::incomplete_branch);
    }

    {
        pvec parents{npos, 0, 1};

        // With cable soma: one cable with 3 samples.
        auto tree = make_tree(parents);
        auto bc = arb::impl::branches_from_parent_index(parents, tree.properties(), false);
        EXPECT_EQ(1u, bc.size());
        EXPECT_EQ(mb({0,1,2},npos), bc[0]);

        // With spherical soma: one sphere and a 2-segment cable.
        // The cable branch is attached to the sphere (i.e. the sphere is the parent branch).
        auto bs = arb::impl::branches_from_parent_index(parents, tree.properties(), true);
        EXPECT_EQ(2u, bs.size());
        EXPECT_EQ(mb({0},npos), bs[0]);
        EXPECT_EQ(mb({1,2},0), bs[1]);
    }

    {
        pvec parents{npos, 0, 0};
        auto tree = make_tree(parents);

        // A spherical root is not valid: each cable branch would have only one sample.
        EXPECT_THROW(arb::impl::branches_from_parent_index(parents, tree.properties(), true), arb::incomplete_branch);

        // Two cables, with two samples each, with the first sample in each being the root
        auto bc = arb::impl::branches_from_parent_index(parents, tree.properties(), false);
        EXPECT_EQ(2u, bc.size());
        EXPECT_EQ(mb({0,1},npos), bc[0]);
        EXPECT_EQ(mb({0,2},npos), bc[1]);
    }

    {
        pvec parents{npos, 0, 1, 2};
        auto tree = make_tree(parents);

        // With cable soma: one cable with 4 samples.
        auto bc = arb::impl::branches_from_parent_index(parents, tree.properties(), false);
        EXPECT_EQ(1u, bc.size());
        EXPECT_EQ(mb({0,1,2,3},npos), bc[0]);

        // With spherical soma: one sphere and on 3-segment cable.
        // The cable branch is attached to the sphere (i.e. the sphere is the parent branch).
        auto bs = arb::impl::branches_from_parent_index(parents, tree.properties(), true);
        EXPECT_EQ(2u, bs.size());
        EXPECT_EQ(mb({0},npos), bs[0]);
        EXPECT_EQ(mb({1,2,3},0), bs[1]);
    }

    {
        pvec parents{npos, 0, 1, 0};
        auto tree = make_tree(parents);

        // With cable soma: two cables with 3 and 2 samples respectively.
        auto bc = arb::impl::branches_from_parent_index(parents, tree.properties(), false);
        EXPECT_EQ(2u, bc.size());
        EXPECT_EQ(mb({0,1,2},npos), bc[0]);
        EXPECT_EQ(mb({0,3},npos), bc[1]);

        // A spherical root is not valid: the second cable branch would have only one sample.
        EXPECT_THROW(arb::impl::branches_from_parent_index(parents, tree.properties(), true), arb::incomplete_branch);
    }

    {
        pvec parents{npos, 0, 1, 0, 3};
        auto tree = make_tree(parents);

        // With cable soma: two cables with 3 samples each [0,1,2] and [0,3,4]
        auto bc = arb::impl::branches_from_parent_index(parents, tree.properties(), false);
        EXPECT_EQ(2u, bc.size());
        EXPECT_EQ(mb({0,1,2},npos), bc[0]);
        EXPECT_EQ(mb({0,3,4},npos), bc[1]);

        // With spherical soma: one sphere and 2 2-sample cables.
        // each of the cable branches is attached to the sphere (i.e. the sphere is the parent branch).
        auto bs = arb::impl::branches_from_parent_index(parents, tree.properties(), true);
        EXPECT_EQ(3u, bs.size());
        EXPECT_EQ(mb({0},npos), bs[0]);
        EXPECT_EQ(mb({1,2},0),  bs[1]);
        EXPECT_EQ(mb({3,4},0),  bs[2]);
    }

    {
        pvec parents{npos, 0, 1, 0, 3, 4, 4, 6};
        auto tree = make_tree(parents);

        // With cable soma: 4 cables: [0,1,2] [0,3,4] [4,5] [4,6,7]
        auto bc = arb::impl::branches_from_parent_index(parents, tree.properties(), false);
        EXPECT_EQ(4u, bc.size());
        EXPECT_EQ(mb({0,1,2},npos), bc[0]);
        EXPECT_EQ(mb({0,3,4},npos), bc[1]);
        EXPECT_EQ(mb({4,5},  1),    bc[2]);
        EXPECT_EQ(mb({4,6,7},1),    bc[3]);

        // With spherical soma: 1 sphere and 4 cables: [1,2] [3,4] [4,5] [4,6,7]
        auto bs = arb::impl::branches_from_parent_index(parents, tree.properties(), true);
        EXPECT_EQ(5u, bs.size());
        EXPECT_EQ(mb({0},   npos), bs[0]);
        EXPECT_EQ(mb({1,2}, 0),    bs[1]);
        EXPECT_EQ(mb({3,4}, 0),    bs[2]);
        EXPECT_EQ(mb({4,5}, 2),    bs[3]);
        EXPECT_EQ(mb({4,6,7}, 2),  bs[4]);
    }
}

// For different parent index vectors, attempt multiple valid and invalid sample sets.
TEST(morphology, construction) {
    constexpr auto npos = arb::mnpos;
    using arb::util::make_span;
    using ms = arb::msample;
    using pvec = std::vector<arb::msize_t>;
    {
        pvec p = {npos, 0};
        std::vector<ms> s = {
            {{0.0, 0.0, 0.0, 1.0}, 1},
            {{0.0, 0.0, 1.0, 1.0}, 1} };

        arb::sample_tree sm(s, p);
        auto m = arb::morphology(sm);

        EXPECT_EQ(1u, m.num_branches());
    }
    {
        pvec p = {npos, 0, 1};
        { // 2-segment cable (1 seg soma, 1 seg dendrite)
            std::vector<ms> s = {
                {{0.0, 0.0, 0.0, 5.0}, 1},
                {{0.0, 0.0, 5.0, 1.0}, 1},
                {{0.0, 0.0, 8.0, 1.0}, 2} };

            arb::sample_tree sm(s, p);
            auto m = arb::morphology(sm);

            EXPECT_EQ(1u, m.num_branches());
        }
        { // spherical soma and single-segment cable
            std::vector<ms> s = {
                {{0.0, 0.0, 0.0, 5.0}, 1},
                {{0.0, 0.0, 1.0, 1.0}, 2},
                {{0.0, 0.0, 8.0, 1.0}, 2} };

            arb::sample_tree sm(s, p);
            auto m = arb::morphology(sm);

            EXPECT_EQ(2u, m.num_branches());
        }
    }
    {
        //              0       |
        //            1   3     |
        //          2           |
        pvec p = {npos, 0, 1, 0};
        {
            // two cables: 1x2 segments, 1x1 segment.
            std::vector<ms> s = {
                {{0.0, 0.0, 0.0, 5.0}, 1},
                {{0.0, 0.0, 5.0, 1.0}, 1},
                {{0.0, 0.0, 6.0, 1.0}, 2},
                {{0.0, 4.0, 0.0, 1.0}, 1}};

            arb::sample_tree sm(s, p);
            auto m = arb::morphology(sm);

            EXPECT_EQ(2u, m.num_branches());
        }
        {
            // error: spherical soma with a single point cable attached via sample 3
            std::vector<ms> s = {
                {{0.0, 0.0, 0.0, 5.0}, 1},
                {{0.0, 0.0, 5.0, 1.0}, 2},
                {{0.0, 0.0, 8.0, 1.0}, 2},
                {{0.0, 5.0, 0.0, 1.0}, 2}};

            arb::sample_tree sm(s, p);
            EXPECT_THROW((arb::morphology(sm)), arb::incomplete_branch);
        }
    }
    {
        //              0       |
        //            1   3     |
        //          2       4    |
        pvec p = {npos, 0, 1, 0, 3};
        {
            // two cables: 1x2 segments, 1x1 segment.
            std::vector<ms> s = {
                {{0.0, 0.0, 0.0, 5.0}, 1},
                {{0.0, 0.0, 5.0, 1.0}, 2},
                {{0.0, 0.0, 8.0, 1.0}, 2},
                {{0.0, 5.0, 0.0, 1.0}, 2},
                {{0.0, 8.0, 0.0, 1.0}, 2}};

            arb::sample_tree sm(s, p);
            auto m = arb::morphology(sm);

            EXPECT_EQ(3u, m.num_branches());
        }
    }
}

// test that morphology generates branch child-parent structure correctly.
TEST(morphology, branches) {
    using pvec = std::vector<arb::msize_t>;
    using svec = std::vector<arb::msample>;
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
        // 0
        pvec parents = {npos};
        svec samples = {
            {{ 0,0,0,3}, 1},
        };
        arb::sample_tree sm(samples, parents);
        arb::morphology m(sm);

        EXPECT_EQ(1u, m.num_branches());
        EXPECT_EQ(npos, m.branch_parent(0));
        EXPECT_EQ(pvec{}, m.branch_children(0));

        check_terminal_branches(m);
    }
    {
        // 0 - 1
        pvec parents = {npos, 0};
        svec samples = {
            {{ 0,0,0,3}, 1},
            {{ 10,0,0,3}, 1},
        };
        arb::sample_tree sm(samples, parents);
        arb::morphology m(sm);

        EXPECT_EQ(1u, m.num_branches());
        EXPECT_EQ(npos, m.branch_parent(0));
        EXPECT_EQ(pvec{}, m.branch_children(0));

        check_terminal_branches(m);
    }
    {
        // 0 - 1 - 2
        pvec parents = {npos, 0, 1};
        {
            // All samples have same tag -> the morphology is a single unbranched cable.
            svec samples = {
                {{ 0,0,0,3}, 1},
                {{10,0,0,3}, 1},
                {{100,0,0,3}, 1},
            };
            arb::sample_tree sm(samples, parents);
            arb::morphology m(sm);

            EXPECT_EQ(1u, m.num_branches());
            EXPECT_EQ(npos, m.branch_parent(0));
            EXPECT_EQ(pvec{}, m.branch_children(0));

            check_terminal_branches(m);
        }
        {
            // First sample has unique tag -> spherical soma attached to a single-segment cable.
            svec samples = {
                {{  0,0,0,10}, 1},
                {{ 10,0,0, 3}, 3},
                {{100,0,0, 3}, 3},
            };
            arb::sample_tree sm(samples, parents);
            arb::morphology m(sm);

            EXPECT_EQ(2u, m.num_branches());
            EXPECT_EQ(npos, m.branch_parent(0));
            EXPECT_EQ(0u,   m.branch_parent(1));
            EXPECT_EQ(pvec{1}, m.branch_children(0));
            EXPECT_EQ(pvec{},  m.branch_children(1));

            check_terminal_branches(m);
        }
    }
    {
        // 2 - 0 - 1
        pvec parents = {npos, 0, 0};

        svec samples = {
            {{  0, 0,0, 5}, 3},
            {{ 10, 0,0, 5}, 3},
            {{  0,10,0, 5}, 3},
        };
        arb::sample_tree sm(samples, parents);
        arb::morphology m(sm);

        EXPECT_EQ(2u, m.num_branches());
        EXPECT_EQ(npos, m.branch_parent(0));
        EXPECT_EQ(npos,   m.branch_parent(1));
        EXPECT_EQ(pvec{}, m.branch_children(0));
        EXPECT_EQ(pvec{},  m.branch_children(1));

        check_terminal_branches(m);
    }
    {
        // 0 - 1 - 2 - 3
        pvec parents = {npos, 0, 1, 2};
    }
    {
        //              0       |
        //             / \      |
        //            1   3     |
        //           /          |
        //          2           |
        pvec parents = {npos, 0, 1, 0};
    }
    {
        // Eight samples
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
        pvec parents = {npos, 0, 1, 0, 3, 4, 4, 6};
        {
            svec samples = {
                {{  0,  0,  0, 10}, 1},
                {{ 10,  0,  0,  2}, 3},
                {{100,  0,  0,  2}, 3},
                {{  0, 10,  0,  2}, 3},
                {{  0,100,  0,  2}, 3},
                {{100,100,  0,  2}, 3},
                {{  0,200,  0,  2}, 3},
                {{  0,300,  0,  2}, 3},
            };
            arb::sample_tree sm(samples, parents);
            arb::morphology m(sm);

            EXPECT_EQ(5u, m.num_branches());
            EXPECT_EQ(npos, m.branch_parent(0));
            EXPECT_EQ(0u,   m.branch_parent(1));
            EXPECT_EQ(0u,   m.branch_parent(2));
            EXPECT_EQ(2u,   m.branch_parent(3));
            EXPECT_EQ(2u,   m.branch_parent(4));
            EXPECT_EQ((pvec{1,2}), m.branch_children(0));
            EXPECT_EQ((pvec{}),    m.branch_children(1));
            EXPECT_EQ((pvec{3,4}), m.branch_children(2));
            EXPECT_EQ((pvec{}),    m.branch_children(3));
            EXPECT_EQ((pvec{}),    m.branch_children(4));

            check_terminal_branches(m);
        }
        {
            svec samples = {
                {{  0,  0,  0, 10}, 3},
                {{ 10,  0,  0,  2}, 3},
                {{100,  0,  0,  2}, 3},
                {{  0, 10,  0,  2}, 3},
                {{  0,100,  0,  2}, 3},
                {{100,100,  0,  2}, 3},
                {{  0,200,  0,  2}, 3},
                {{  0,300,  0,  2}, 3},
            };
            arb::sample_tree sm(samples, parents);
            arb::morphology m(sm);

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
}

TEST(morphology, swc) {
    std::string datadir{DATADIR};
    auto fname = datadir + "/example.swc";
    std::ifstream fid(fname);
    if (!fid.is_open()) {
        std::cerr << "unable to open file " << fname << "... skipping test\n";
        return;
    }

    // Load swc samples from file.
    auto swc_samples = arb::parse_swc_file(fid);

    // Build a sample_tree from swc samples.
    auto sm = arb::swc_as_sample_tree(swc_samples);
    EXPECT_EQ(1058u, sm.size()); // file contains 195 samples

    // Test that the morphology contains the expected number of branches.
    auto m = arb::morphology(sm);
    EXPECT_EQ(31u, m.num_branches());
}

TEST(morphology, minset) {
    using pvec = std::vector<arb::msize_t>;
    using svec = std::vector<arb::msample>;
    using ll = arb::mlocation_list;
    constexpr auto npos = arb::mnpos;

    // Eight samples
    //                  no-sphere  sphere
    //          sample   branch    branch
    //            0         0         0
    //           1 3       0 1       1 2
    //          2   4     0   1     1   2
    //             5 6       2 3       3 4
    //                7         3         4
    pvec parents = {npos, 0, 1, 0, 3, 4, 4, 6};
    svec samples = {
        {{  0,  0,  0,  2}, 3},
        {{ 10,  0,  0,  2}, 3},
        {{100,  0,  0,  2}, 3},
        {{  0, 10,  0,  2}, 3},
        {{  0,100,  0,  2}, 3},
        {{100,100,  0,  2}, 3},
        {{  0,200,  0,  2}, 3},
        {{  0,300,  0,  2}, 3},
    };
    arb::sample_tree sm(samples, parents);
    {
        arb::morphology m(sm, false);

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
    {
        arb::morphology m(sm, true);

        EXPECT_EQ((ll{}), minset(m, ll{}));
        EXPECT_EQ((ll{{2,0.1}}), minset(m, ll{{2,0.1}}));
        EXPECT_EQ((ll{{0,0.5}}), minset(m, ll{{0,0.5}, {1,0.5}}));
        EXPECT_EQ((ll{{0,0.5}}), minset(m, ll{{0,0.5}}));
        EXPECT_EQ((ll{{0,0}}), minset(m, ll{{0,0}, {0,0.5}, {1,0}, {1,0.5}}));
        EXPECT_EQ((ll{{1,0.5}, {3,0.1}, {4,0.5}}), minset(m, ll{{1,0.5}, {1,1}, {3,0.1}, {4,0.5}, {4,0.7}}));
    }
}

// Testing mextent; intersection and join operations are
// exercised by region/locset thingifies in test_morph_expr.cpp.

TEST(mextent, invariants) {
    using namespace arb;
    using testing::cablelist_eq;

    using pvec = std::vector<msize_t>;
    using svec = std::vector<msample>;
    using cl = mcable_list;

    // Eight samples
    //                  no-sphere
    //          sample   branch
    //            0         0
    //           1 3       0 1
    //          2   4     0   1
    //             5 6       2 3
    //                7         3
    pvec parents = {mnpos, 0, 1, 0, 3, 4, 4, 6};
    svec samples = {
        {{  0,  0,  0,  2}, 3},
        {{ 10,  0,  0,  2}, 3},
        {{100,  0,  0,  2}, 3},
        {{  0, 10,  0,  2}, 3},
        {{  0,100,  0,  2}, 3},
        {{100,100,  0,  2}, 3},
        {{  0,200,  0,  2}, 3},
        {{  0,300,  0,  2}, 3},
    };

    morphology m(sample_tree(samples, parents));

    mextent x1(m, cl{{1, 0.25, 0.75}});
    ASSERT_TRUE(x1.test_invariants(m));
    EXPECT_TRUE(cablelist_eq(cl{{1, 0.25, 0.75}}, x1.cables()));

    mextent x2(m, cl{{1, 0., 0.75}});
    ASSERT_TRUE(x2.test_invariants(m));
    EXPECT_TRUE(cablelist_eq(cl{{0, 0, 0}, {1, 0., 0.75}}, x2.cables()));

    mextent x3(m, cl{{2, 0., 1.}});
    ASSERT_TRUE(x3.test_invariants(m));
    EXPECT_TRUE(cablelist_eq(cl{{1, 1, 1}, {2, 0., 1.}, {3, 0., 0.}}, x3.cables()));
}

TEST(mextent, intersects) {
    using namespace arb;
    using testing::cablelist_eq;

    using pvec = std::vector<msize_t>;
    using svec = std::vector<msample>;
    using cl = mcable_list;

    // Eight samples
    //                  no-sphere
    //          sample   branch
    //            0         0
    //           1 3       0 1
    //          2   4     0   1
    //             5 6       2 3
    //                7         3
    pvec parents = {mnpos, 0, 1, 0, 3, 4, 4, 6};
    svec samples = {
        {{  0,  0,  0,  2}, 3},
        {{ 10,  0,  0,  2}, 3},
        {{100,  0,  0,  2}, 3},
        {{  0, 10,  0,  2}, 3},
        {{  0,100,  0,  2}, 3},
        {{100,100,  0,  2}, 3},
        {{  0,200,  0,  2}, 3},
        {{  0,300,  0,  2}, 3},
    };

    morphology m(sample_tree(samples, parents));

    mextent x1(m, cl{{1, 0.25, 0.75}});
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

    mextent x2(m, cl{{1, 0., 0.75}});
    EXPECT_TRUE(x2.intersects(mlocation{1, 0.}));
    EXPECT_TRUE(x2.intersects(mlocation{0, 0.}));
    EXPECT_FALSE(x2.intersects(mlocation{0, 1.}));

    mextent x3(m, cl{{2, 0., 1.}});
    EXPECT_FALSE(x3.intersects(mlocation{1, 0.}));
    EXPECT_TRUE(x3.intersects(mlocation{1, 1.}));
    EXPECT_TRUE(x3.intersects(mlocation{3, 0.}));
    EXPECT_FALSE(x3.intersects(mlocation{3, 1.}));
}

TEST(mextent, canonical) {
    using namespace arb;
    using testing::cablelist_eq;

    using pvec = std::vector<msize_t>;
    using svec = std::vector<msample>;
    using cl = mcable_list;

    // Eight samples
    //                  no-sphere
    //          sample   branch
    //            0         0
    //           1 3       0 1
    //          2   4     0   1
    //             5 6       2 3
    //                7         3
    pvec parents = {mnpos, 0, 1, 0, 3, 4, 4, 6};
    svec samples = {
        {{  0,  0,  0,  2}, 3},
        {{ 10,  0,  0,  2}, 3},
        {{100,  0,  0,  2}, 3},
        {{  0, 10,  0,  2}, 3},
        {{  0,100,  0,  2}, 3},
        {{100,100,  0,  2}, 3},
        {{  0,200,  0,  2}, 3},
        {{  0,300,  0,  2}, 3},
    };

    morphology m(sample_tree(samples, parents));

    mextent x1(m, cl{{0, 0.3, 0.7}, {1, 0.5, 0.7}, {3, 0.4, 1}});
    EXPECT_TRUE(cablelist_eq(canonical(m, x1), x1.cables()));

    mcable_list cl2{{1, 0.5, 1}, {2, 0, 0.5}}; // a reduced representation
    mextent x2(m, cl2);
    EXPECT_FALSE(cablelist_eq(cl2, x2.cables()));
    EXPECT_TRUE(cablelist_eq(cl2, canonical(m, x2)));

    mcable_list cl3{{3, 0, 0}}; // extent is cover of fork point
    mextent x3(m, cl3);
    EXPECT_TRUE(cablelist_eq(cl{{1, 1, 1}}, canonical(m, x3)));

    mcable_list cl4{{1, 1, 1}, {3, 0, 0.5}}; // canonical repn excludes zero-length cable
    mextent x4(m, cl4);
    EXPECT_TRUE(cablelist_eq(cl{{3, 0, 0.5}}, canonical(m, x4)));

    // Heads of top-level branches should be preserved.
    mcable_list cl5{{1, 0, 0}};
    mextent x5(m, cl5);
    EXPECT_TRUE(cablelist_eq(cl{{0, 0, 0}, {1, 0, 0}}, x5.cables()));
    EXPECT_TRUE(cablelist_eq(x5.cables(), canonical(m, x5)));
}
