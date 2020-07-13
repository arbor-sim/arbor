#include <algorithm>
#include <fstream>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "../test/gtest.h"

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

