/*
 * Unit tests for em_morphology, region, locset, label_dict.
 */

#include <cmath>
#include <string>
#include <vector>

#include "../test/gtest.h"

#include <arbor/math.hpp>
#include <arbor/morph/error.hpp>
#include <arbor/morph/morphology.hpp>
#include <arbor/morph/sample_tree.hpp>

#include "io/sepval.hpp"
#include "morph/morphology_impl.hpp"
#include "morph/em_morphology.hpp"

template <typename T>
std::ostream& operator<<(std::ostream& o, const std::vector<T>& v) {
    return o << "[" << arb::io::csv(v) << "]";
}

TEST(em_morphology, locations) {
    using pvec = std::vector<arb::msize_t>;
    using svec = std::vector<arb::msample>;
    using loc = arb::mlocation;
    constexpr auto npos = arb::mnpos;

    {
        pvec parents = {npos, 0, 1, 2, 3};
        svec samples = {
            {{  0,  0,  0,  2}, 1},
            {{  1,  0,  0,  2}, 1},
            {{  3,  0,  0,  2}, 1},
            {{  7,  0,  0,  2}, 1},
            {{ 10,  0,  0,  2}, 1},
        };
        arb::sample_tree sm(samples, parents);
        arb::em_morphology em(arb::morphology(sm, false));
        EXPECT_EQ(em.sample2loc(0), (loc{0,0}));
        EXPECT_EQ(em.sample2loc(1), (loc{0,0.1}));
        EXPECT_EQ(em.sample2loc(2), (loc{0,0.3}));
        EXPECT_EQ(em.sample2loc(3), (loc{0,0.7}));
        EXPECT_EQ(em.sample2loc(4), (loc{0,1}));

        EXPECT_EQ(em.root(),      (loc{0,0}));
        EXPECT_EQ(em.terminals(), (arb::mlocation_list{{0,1}}));
    }

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
    {   // Spherical root.
        pvec parents = {npos, 0, 1, 0, 3, 4, 4, 6};

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
        arb::em_morphology em(arb::morphology(sm, true));

        EXPECT_EQ(em.sample2loc(0), (loc{0,0}));
        EXPECT_EQ(em.sample2loc(1), (loc{1,0}));
        EXPECT_EQ(em.sample2loc(2), (loc{1,1}));
        EXPECT_EQ(em.sample2loc(3), (loc{2,0}));
        EXPECT_EQ(em.sample2loc(4), (loc{2,1}));
        EXPECT_EQ(em.sample2loc(5), (loc{3,1}));
        EXPECT_EQ(em.sample2loc(6), (loc{4,0.5}));
        EXPECT_EQ(em.sample2loc(7), (loc{4,1}));

        EXPECT_EQ(em.root(),      (loc{0,0}));
        EXPECT_EQ(em.terminals(), (arb::mlocation_list{{1,1}, {3,1}, {4,1}}));
    }
    {   // No Spherical root
        pvec parents = {npos, 0, 1, 0, 3, 4, 4, 6};

        svec samples = {
            {{  0,  0,  0,  2}, 1},
            {{ 10,  0,  0,  2}, 3},
            {{100,  0,  0,  2}, 3},
            {{  0, 10,  0,  2}, 3},
            {{  0,100,  0,  2}, 3},
            {{100,100,  0,  2}, 3},
            {{  0,130,  0,  2}, 3},
            {{  0,300,  0,  2}, 3},
        };
        arb::sample_tree sm(samples, parents);
        arb::em_morphology em(arb::morphology(sm, false));

        EXPECT_EQ(em.sample2loc(0), (loc{0,0}));
        EXPECT_EQ(em.sample2loc(1), (loc{0,0.1}));
        EXPECT_EQ(em.sample2loc(2), (loc{0,1}));
        EXPECT_EQ(em.sample2loc(3), (loc{1,0.1}));
        EXPECT_EQ(em.sample2loc(4), (loc{1,1}));
        EXPECT_EQ(em.sample2loc(5), (loc{2,1}));
        EXPECT_EQ(em.sample2loc(6), (loc{3,0.15}));
        EXPECT_EQ(em.sample2loc(7), (loc{3,1}));

        EXPECT_EQ(em.root(),      (loc{0,0}));
        EXPECT_EQ(em.terminals(), (arb::mlocation_list{{0,1}, {2,1}, {3,1}}));
    }
}

TEST(em_morphology, locset) {
    
}
