#include <algorithm>
#include <fstream>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <arbor/morph/morphexcept.hpp>

#include "arbor/morph/primitives.hpp"
#include "util/span.hpp"

#include "morph_pred.hpp"

// Test basic functions on properties of mpoint
TEST(segment_tree, mpoint) {
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

TEST(segment_tree, empty) {
    using mp = arb::mpoint;
    arb::segment_tree tree;
    EXPECT_TRUE(tree.empty());
    tree.append(arb::mnpos, mp{0,0,0,1}, mp{0,0,1,1}, 1);
    EXPECT_FALSE(tree.empty());
}

TEST(segment_tree, invalid_append) {
    using mp = arb::mpoint;
    using arb::mnpos;
    arb::segment_tree tree;

    // Test that appropriate exceptions are thrown when attempting to create
    // a segment by implicitly extending from the root.
    EXPECT_THROW(tree.append(mnpos, mp{0,0,0,1}, 1), arb::invalid_segment_parent);

    tree.append(mnpos, mp{0,0,0,1}, mp{0,0,1,1}, 1);

    EXPECT_THROW(tree.append(mnpos, mp{0,0,0,1}, 1), arb::invalid_segment_parent);

    // Test that an exception is thrown when attempting to append to
    // a segment that is not in the tree (parent>=num_segs)
    EXPECT_THROW(tree.append(1, mp{0,0,0,1}, mp{0,0,1,1}, 1), arb::invalid_segment_parent);
    EXPECT_THROW(tree.append(2, mp{0,0,0,1}, mp{0,0,1,1}, 1), arb::invalid_segment_parent);
    EXPECT_THROW(tree.append(2, mp{0,0,1,1}, 1), arb::invalid_segment_parent);
}

// Generate some random morphologies of different sizes, and verify that
// the correct tree is constructed.
TEST(segment_tree, fuzz) {
    using mp = arb::mpoint;
    using arb::mnpos;

    int nrun = 10;
    int max_size = 1<<12;

    std::mt19937 gen(0);

    auto make_point = [](arb::msize_t i) {return mp{0,0,double(i),1};};

    for (auto nseg=2; nseg<=max_size; nseg*=2) {
        for (int run=0; run<nrun; ++run) {
            arb::segment_tree tree;
            std::vector<arb::msize_t> parents;

            parents.reserve(nseg);
            tree.reserve(nseg);
            for (int i=0; i<nseg; ++i) {
                std::uniform_int_distribution<int> distrib(-1, i-1);

                // Draw random segment to attach to.
                arb::msize_t p = distrib(gen);
                parents.push_back(p);

                // Attach segment.
                // Use an implicit append every third time when possible.
                if (p!=mnpos && i%3)
                    tree.append(p, make_point(i), i);
                else
                    tree.append(p, make_point(p), make_point(i), i);

                // Validate that the correct number of segments created.
                EXPECT_EQ(tree.size(), std::size_t(i+1));
            }
            EXPECT_EQ(parents, tree.parents());
            for (int i=0; i<nseg; ++i) {
                arb::msegment seg = tree.segments()[i];
                EXPECT_EQ(seg.prox, make_point(parents[i]));
                EXPECT_EQ(seg.dist, make_point(i));
                EXPECT_EQ(seg.tag, i);
            }
        }
    }
}

::testing::AssertionResult trees_equivalent(const arb::segment_tree& a,
                                            const arb::segment_tree& b) {
    if (!arb::equivalent(a, b)) return ::testing::AssertionFailure() << "Trees are not equivalent:\n"
                                                                     << a
                                                                     << "\nand:\n"
                                                                     << b;
    return ::testing::AssertionSuccess();
}

TEST(segment_tree, split) {
    // linear chain
    {
        arb::segment_tree tree;
        tree.append(arb::mnpos, {0, 0, 0}, {0, 0, 1}, 0);
        tree.append(0,          {0, 0, 1}, {0, 0, 2}, 0);
        tree.append(1,          {0, 0, 2}, {0, 0, 3}, 0);
        tree.append(2,          {0, 0, 3}, {0, 0, 4}, 0);
        tree.append(3,          {0, 0, 4}, {0, 0, 5}, 0);
        {
            arb::segment_tree p, q;
            q.append(arb::mnpos, {0, 0, 0}, {0, 0, 1}, 0);
            q.append(0,          {0, 0, 1}, {0, 0, 2}, 0);
            q.append(1,          {0, 0, 2}, {0, 0, 3}, 0);
            q.append(2,          {0, 0, 3}, {0, 0, 4}, 0);
            q.append(3,          {0, 0, 4}, {0, 0, 5}, 0);
            const auto& [l, r] = arb::split_at(tree, 0);
            EXPECT_TRUE(trees_equivalent(p, l));
            EXPECT_TRUE(trees_equivalent(q, r));
        }
        {
            arb::segment_tree p, q;
            p.append(arb::mnpos, {0, 0, 0}, {0, 0, 1}, 0);
            p.append(0,          {0, 0, 1}, {0, 0, 2}, 0);
            q.append(arb::mnpos, {0, 0, 2}, {0, 0, 3}, 0);
            q.append(0,          {0, 0, 3}, {0, 0, 4}, 0);
            q.append(1,          {0, 0, 4}, {0, 0, 5}, 0);
            const auto& [l, r] = arb::split_at(tree, 2);
            EXPECT_TRUE(trees_equivalent(p, l));
            EXPECT_TRUE(trees_equivalent(q, r));
        }
        {
            arb::segment_tree p, q;
            p.append(arb::mnpos, {0, 0, 0}, {0, 0, 1}, 0);
            p.append(0,          {0, 0, 1}, {0, 0, 2}, 0);
            p.append(1,          {0, 0, 2}, {0, 0, 3}, 0);
            p.append(2,          {0, 0, 3}, {0, 0, 4}, 0);
            q.append(arb::mnpos, {0, 0, 4}, {0, 0, 5}, 0);
            const auto& [l, r] = arb::split_at(tree, 4);
            EXPECT_TRUE(trees_equivalent(p, l));
            EXPECT_TRUE(trees_equivalent(q, r));
        }
    }
    // Error cases
    {
        arb::segment_tree t;
        EXPECT_THROW(arb::split_at(t, arb::mnpos), arb::arbor_exception);
        EXPECT_THROW(arb::split_at(t, 1),          arb::arbor_exception);
    }
    /* gnarly tree
    ** (npos) - 0 - 1 - 4
    **            \
    **              2 - 3
    **                \
    **                  5
    */
    // NB: Splitting _will_ re-order segments. So we have to be careful when
    //     building values to compare against.
    {
        arb::segment_tree tree;
        tree.append(arb::mnpos, {0, 0, 0}, {0, 0, 1}, 0); // 0
        tree.append(0,          {0, 0, 1}, {0, 0, 2}, 0); // 1
        tree.append(0,          {0, 0, 2}, {0, 0, 3}, 0); // 2
        tree.append(2,          {0, 0, 3}, {0, 0, 4}, 0); // 3
        tree.append(1,          {0, 0, 4}, {0, 0, 5}, 0); // 4
        tree.append(2,          {0, 0, 5}, {0, 0, 6}, 0); // 5
        {
            arb::segment_tree p, q;

            q.append(arb::mnpos, {0, 0, 0}, {0, 0, 1}, 0);
            q.append(0,          {0, 0, 2}, {0, 0, 3}, 0);
            q.append(1,          {0, 0, 5}, {0, 0, 6}, 0);
            q.append(1,          {0, 0, 3}, {0, 0, 4}, 0);
            q.append(0,          {0, 0, 1}, {0, 0, 2}, 0);
            q.append(4,          {0, 0, 4}, {0, 0, 5}, 0);

            const auto& [l, r] = arb::split_at(tree, 0);

            EXPECT_TRUE(trees_equivalent(p, l));
            EXPECT_TRUE(trees_equivalent(q, r));
        }
        {
            arb::segment_tree p, q;

            p.append(arb::mnpos, {0, 0, 0}, {0, 0, 1}, 0);
            p.append(0,          {0, 0, 2}, {0, 0, 3}, 0);
            p.append(1,          {0, 0, 5}, {0, 0, 6}, 0);
            p.append(1,          {0, 0, 3}, {0, 0, 4}, 0);

            q.append(arb::mnpos, {0, 0, 1}, {0, 0, 2}, 0);
            q.append(0,          {0, 0, 4}, {0, 0, 5}, 0);

            const auto& [l, r] = arb::split_at(tree, 1);
            EXPECT_TRUE(trees_equivalent(p, l));
            EXPECT_TRUE(trees_equivalent(q, r));
        }
        {
            arb::segment_tree p, q;

            p.append(arb::mnpos, {0, 0, 0}, {0, 0, 1}, 0);
            p.append(0,          {0, 0, 1}, {0, 0, 2}, 0);
            p.append(1,          {0, 0, 4}, {0, 0, 5}, 0);

            q.append(arb::mnpos, {0, 0, 2}, {0, 0, 3}, 0);
            q.append(0,          {0, 0, 5}, {0, 0, 6}, 0);
            q.append(0,          {0, 0, 3}, {0, 0, 4}, 0);

            const auto& [l, r] = arb::split_at(tree, 2);
            EXPECT_TRUE(trees_equivalent(p, l));
            EXPECT_TRUE(trees_equivalent(q, r));
        }
    }
}

TEST(segment_tree, join) {
    // linear chain
    {
        arb::segment_tree tree;
        tree.append(arb::mnpos, {0, 0, 0}, {0, 0, 1}, 0);
        tree.append(0,          {0, 0, 1}, {0, 0, 2}, 0);
        tree.append(1,          {0, 0, 2}, {0, 0, 3}, 0);
        tree.append(2,          {0, 0, 3}, {0, 0, 4}, 0);
        tree.append(3,          {0, 0, 4}, {0, 0, 5}, 0);
        {
            arb::segment_tree p, q;

            q.append(arb::mnpos, {0, 0, 0}, {0, 0, 1}, 0);
            q.append(0,          {0, 0, 1}, {0, 0, 2}, 0);
            q.append(1,          {0, 0, 2}, {0, 0, 3}, 0);
            q.append(2,          {0, 0, 3}, {0, 0, 4}, 0);
            q.append(3,          {0, 0, 4}, {0, 0, 5}, 0);

            const auto& t = arb::join_at(p, arb::mnpos, q);
            EXPECT_TRUE(trees_equivalent(tree, t));
        }
        {
            arb::segment_tree p, q;

            p.append(arb::mnpos, {0, 0, 0}, {0, 0, 1}, 0);
            p.append(0,          {0, 0, 1}, {0, 0, 2}, 0);

            q.append(arb::mnpos, {0, 0, 2}, {0, 0, 3}, 0);
            q.append(0,          {0, 0, 3}, {0, 0, 4}, 0);
            q.append(1,          {0, 0, 4}, {0, 0, 5}, 0);

            const auto& t = arb::join_at(p, 1, q);
            EXPECT_TRUE(trees_equivalent(tree, t));
        }
    }

    // Error cases
    {
        arb::segment_tree t;
        EXPECT_THROW(arb::split_at(t, arb::mnpos), arb::arbor_exception);
        EXPECT_THROW(arb::split_at(t, 1),          arb::arbor_exception);
    }
    /* gnarly tree
    ** (npos) - 0 - 1 - 4
    **            \
    **              2 - 3
    **                \
    **                  5
    */
    // NB: Joining _will_ re-order segments. So we have to be careful when
    //     building values to compare against.
    {
        arb::segment_tree tree;
        tree.append(arb::mnpos, {0, 0, 0}, {0, 0, 1}, 0); // 0
        tree.append(0,          {0, 0, 1}, {0, 0, 2}, 0); // 1
        tree.append(0,          {0, 0, 2}, {0, 0, 3}, 0); // 2
        tree.append(2,          {0, 0, 3}, {0, 0, 4}, 0); // 3
        tree.append(1,          {0, 0, 4}, {0, 0, 5}, 0); // 4
        tree.append(2,          {0, 0, 5}, {0, 0, 6}, 0); // 5
        {
            arb::segment_tree p, q;

            q.append(arb::mnpos, {0, 0, 0}, {0, 0, 1}, 0);
            q.append(0,          {0, 0, 2}, {0, 0, 3}, 0);
            q.append(1,          {0, 0, 5}, {0, 0, 6}, 0);
            q.append(1,          {0, 0, 3}, {0, 0, 4}, 0);
            q.append(0,          {0, 0, 1}, {0, 0, 2}, 0);
            q.append(4,          {0, 0, 4}, {0, 0, 5}, 0);

            const auto& t = arb::join_at(p, arb::mnpos, q);
            EXPECT_TRUE(trees_equivalent(tree, t));
        }
        {
            arb::segment_tree p, q;

            p.append(arb::mnpos, {0, 0, 0}, {0, 0, 1}, 0);
            p.append(0,          {0, 0, 2}, {0, 0, 3}, 0);
            p.append(1,          {0, 0, 5}, {0, 0, 6}, 0);
            p.append(1,          {0, 0, 3}, {0, 0, 4}, 0);

            q.append(arb::mnpos, {0, 0, 1}, {0, 0, 2}, 0);
            q.append(0,          {0, 0, 4}, {0, 0, 5}, 0);

            const auto& t = arb::join_at(p, 0, q);
            EXPECT_TRUE(trees_equivalent(tree, t));
        }
        {
            arb::segment_tree p, q;

            p.append(arb::mnpos, {0, 0, 0}, {0, 0, 1}, 0);
            p.append(0,          {0, 0, 1}, {0, 0, 2}, 0);
            p.append(1,          {0, 0, 4}, {0, 0, 5}, 0);

            q.append(arb::mnpos, {0, 0, 2}, {0, 0, 3}, 0);
            q.append(0,          {0, 0, 5}, {0, 0, 6}, 0);
            q.append(0,          {0, 0, 3}, {0, 0, 4}, 0);

            const auto& t = arb::join_at(p, 0, q);
            EXPECT_TRUE(trees_equivalent(tree, t));
        }
    }
}


TEST(segment_tree, tag_roots) {
    // linear chain
    {
        arb::segment_tree tree;
        tree.append(arb::mnpos, {0, 0, 0}, {0, 0, 1}, 1);
        tree.append(0,          {0, 0, 1}, {0, 0, 2}, 1);
        tree.append(1,          {0, 0, 2}, {0, 0, 3}, 2);
        tree.append(2,          {0, 0, 3}, {0, 0, 4}, 3);
        tree.append(3,          {0, 0, 4}, {0, 0, 5}, 2);
        {
            EXPECT_EQ((std::vector<arb::msize_t>{0}), arb::tag_roots(tree, 1));
            EXPECT_EQ((std::vector<arb::msize_t>{2, 4}), arb::tag_roots(tree, 2));
            EXPECT_EQ((std::vector<arb::msize_t>{3}), arb::tag_roots(tree, 3));
        }
    }
    /* gnarly tree
    ** (npos) - 0 - 1 - 4
    **            \
    **              2 - 3
    **                \
    **                 5
    */
    {
        arb::segment_tree tree;
        tree.append(arb::mnpos, {0, 0, 0}, {0, 0, 1}, 1); // 0
        tree.append(0,          {0, 0, 1}, {0, 0, 2}, 3); // 1
        tree.append(0,          {0, 0, 2}, {0, 0, 3}, 2); // 2
        tree.append(2,          {0, 0, 3}, {0, 0, 4}, 5); // 3
        tree.append(1,          {0, 0, 4}, {0, 0, 5}, 4); // 4
        tree.append(2,          {0, 0, 5}, {0, 0, 6}, 5); // 5
        {
            EXPECT_EQ((std::vector<arb::msize_t>{0}), arb::tag_roots(tree, 1));
            EXPECT_EQ((std::vector<arb::msize_t>{2}), arb::tag_roots(tree, 2));
            EXPECT_EQ((std::vector<arb::msize_t>{1}), arb::tag_roots(tree, 3));
            EXPECT_EQ((std::vector<arb::msize_t>{4}), arb::tag_roots(tree, 4));
            EXPECT_EQ((std::vector<arb::msize_t>{3, 5}), arb::tag_roots(tree, 5));
        }
    }
}
